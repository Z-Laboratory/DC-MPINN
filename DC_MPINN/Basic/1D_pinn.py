import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class PINN_Net(nn.Module):
    def __init__(self):
        super(PINN_Net, self).__init__()
        a = 100 #increasing this is a great way to inprove accuracy (at the cost of compute time)
        self.hidden_layer1 = nn.Linear(2, a)
        self.hidden_layer2 = nn.Linear(a, a)
        self.hidden_layer3 = nn.Linear(a, a)
        self.hidden_layer4 = nn.Linear(a, a)
        # self.hidden_layer5 = nn.Linear(a, a)
        # self.hidden_layer6 = nn.Linear(a, a)
        # self.hidden_layer7 = nn.Linear(a, a)
        # self.hidden_layer8 = nn.Linear(a, a)
        self.output_layer = nn.Linear(a, 4)
        #self.activation = torch.relu
        #self.activation=torch.nn.LeakyReLU()
        self.activation=torch.tanh
        '''
        relu is much more erratic, but produces better absolute results faster
        tanh and sigmoid give a stable lower bound, but are slower
        '''

    def forward(self,t,x):
        inputs = torch.cat([t,x], axis=1)  # combined two arrays of 1 columns each to one array of 2 columns
        layer1_out = self.activation(self.hidden_layer1(inputs))
        layer2_out = self.activation(self.hidden_layer2(layer1_out))
        layer3_out = self.activation(self.hidden_layer3(layer2_out))
        layer4_out = self.activation(self.hidden_layer4(layer3_out))
        output = self.output_layer(layer4_out)
        return output

class DEM_Net(nn.Module):
    def __init__(self):
        super(DEM_Net, self).__init__()
        a = 30 #increasing this is a great way to inprove accuracy (at the cost of compute time)
        self.hidden_layer1 = nn.Linear(1, a)
        self.hidden_layer2 = nn.Linear(a, a)
        # self.hidden_layer3 = nn.Linear(a, a)
        # self.hidden_layer4 = nn.Linear(a, a)
        self.output_layer = nn.Linear(a, 1)
        #self.activation = torch.relu
        #self.activation=torch.nn.LeakyReLU()
        self.activation=torch.tanh

    def forward(self,t):
        #inputs = torch.cat([s1,x, y, z], axis=1)  # combined two arrays of 1 columns each to one array of 2 columns
        layer1_out = self.activation(self.hidden_layer1(t))
        layer2_out = self.activation(self.hidden_layer2(layer1_out))
        # layer3_out = self.activation(self.hidden_layer3(layer2_out))
        # layer4_out = self.activation(self.hidden_layer4(layer3_out))
        output = self.output_layer(layer2_out)
        return output

class method_comparision():
    def __init__(self,m,k,l,f):
        self.m=m
        self.k=k
        self.f=f
        self.l=l
        self.times=np.linspace(0,1,101)
    def loss_sum(self, tinput):
        return torch.sum(tinput) / tinput.data.nelement()

    def analytic(self):
        #newtonian mechanics to solve differential equaiton for balance of forces
        x0=self.f/-self.k
        analytic_solution=self.l+self.f/self.k-self.f/self.k*np.cos(np.sqrt(self.k/self.m)*self.times)
        plt.plot(self.times,analytic_solution,label="Analytic solution")

        omega=np.sqrt(self.k/self.m)
        critical_damping_coefficient=2*self.m*omega
        A=self.f/self.k/2
        B=self.f/self.k/2
        #critial_damped_analytic=self.l+self.f/self.k-A*np.exp(-omega*self.times*(critical_damping_coefficient/(2*self.m*omega)))
        #critial_damped_analytic = self.l + self.f / self.k - A * np.exp(-omega * self.times * 1)

        d=1

        critial_damped_analytic = self.l + self.f / self.k - A * np.exp(-omega * self.times *(d))- B * np.exp(-omega * self.times *(d))
        plt.plot(self.times, critial_damped_analytic, label="Critically damped Analytic solution")

    def DEM(self):
        #deep energy method to apply the principal of least action to the system

        #NOTE: We do not contrain the inital velocity of the system to be 0, as such, we have have different behavior
        #The result here closely mirrors the behavior of a critically damped system (This makes intuitive sense as this is the system that will dissapate energy fastest

        DEM=DEM_Net()
        Dem_optimizer=optimizer = torch.optim.Rprop(DEM.parameters(), lr=.0002, step_sizes=[1e-8, 1])
        iterations=1000
        num_times=1000
        energy=np.zeros(iterations)

        for epoch in range(iterations):
            Dem_optimizer.zero_grad()
            times = torch.rand((num_times, 1), requires_grad=True)
            displacements=torch.flatten(DEM(times))*torch.flatten(times)
            displacements=displacements.unsqueeze(1)
            internal_energy=self.loss_sum(.5*self.k*torch.square(displacements))
            velocities=torch.autograd.grad(displacements,times,torch.ones_like(times), create_graph=True, retain_graph=True)[0]
            kinetic_energy=self.loss_sum(.5*self.m*torch.square(velocities))
            bc_energy=self.loss_sum(1*self.f*displacements)
            total_energy=internal_energy+kinetic_energy-bc_energy
            total_energy.backward()
            Dem_optimizer.step()
            with torch.autograd.no_grad():
                if epoch % 100 == 0:
                    print(epoch, "total_energy:", total_energy.data,"internal",internal_energy.data,"kinetic",kinetic_energy.data,"bc",bc_energy.data)
                    # print(epoch, "Traning Loss:", stress_divergence_loss.data, linear_elasticity_loss.data, volume_loss.data)
                energy[epoch] = total_energy.data


        times=torch.linspace(0,1,101)
        times=torch.reshape(times,(101,1))
        displacements=torch.flatten(DEM(times))*torch.flatten(times)
        plt.plot(times.data.cpu().detach().numpy(),1+displacements.data.cpu().detach().numpy(),label="DEM")
    def PINN(self,end):
        PINN = PINN_Net().to(device)
        PINN_optimizer = torch.optim.Rprop(PINN.parameters(), lr=.0002, step_sizes=[1e-8, 1])
        iterations = 2500
        num_times = 2000
        energy = np.zeros(iterations)
        loss_function=torch.nn.MSELoss()

        for epoch in range(iterations):
            PINN_optimizer.zero_grad()
            end=end
            times = torch.rand((num_times, 1), requires_grad=True).to(device)*end*1.1
            x_values=torch.rand((num_times, 1), requires_grad=True).to(device)*self.l*1.1
            if epoch == iterations-1:
                x_values = torch.linspace(0,self.l,num_times, requires_grad=True)
                x_values = torch.reshape(x_values, (num_times, 1)).to(device)
                times = torch.linspace(0,end,num_times, requires_grad=True)
                #times=torch.ones_like(x_values,requires_grad=True)*end
                times = torch.reshape(times,(num_times,1)).to(device)
                x_values=torch.ones_like(times,requires_grad=True)*.001
            outputs=PINN(times,x_values)
            f=torch.tensor(self.f)
            k=torch.tensor(self.k)
            m=torch.tensor(self.m)

            #################### This seems to highlight the issue with the optimization/epressivity ####################
            p = outputs[:, 0] *torch.flatten(times)* torch.flatten(x_values)                            #loss is substantially higher
            #p = torch.flatten(f/k-f/k*torch.cos(torch.sqrt(k/m)*times)) *torch.flatten(x_values)           #Loss is lower (this is the ground truth)
            v = outputs[:, 1] *torch.flatten(times)* torch.flatten(x_values)
            a = outputs[:, 2] *torch.flatten(x_values)
            s = outputs[:, 3]

            combined=torch.stack((p,v,a,s),dim=1)

            s_ext=p

            #Strain vs test is the major issue here!!!!!!!!!!!!
            test=torch.flatten(p)/torch.flatten(x_values)

            v_ext = torch.autograd.grad(p, times, torch.ones_like(p), create_graph=True, retain_graph=True)[0]
            a_ext2 = torch.autograd.grad(v_ext, times, torch.ones_like(times), create_graph=True, retain_graph=True)[0]
            a_ext = torch.autograd.grad(v, times, torch.ones_like(p), create_graph=True, retain_graph=True)[0]

            v_loss = loss_function(v, torch.flatten(v_ext)) * 10
            a_loss = loss_function(a, torch.flatten(a_ext)) * 1

            fb = self.f * torch.ones_like(p)
            stress_zeros = torch.zeros_like(fb)
            # AAAA=torch.flatten(a_ext2) * self.m
            # BBBB=p * self.k
            actual = torch.flatten(a) * self.m + torch.flatten(test) * self.k - fb*torch.flatten(x_values)


            constitutative=loss_function(actual,stress_zeros)*1

            weighting=torch.tensor([1,epoch/iterations])
            total_energy = constitutative+v_loss+a_loss

            #we need to add some term related to the elastic force working on the system

            total_energy.backward()
            PINN_optimizer.step()
            if epoch==iterations-1:
                inertia=torch.flatten(a)*self.m
                spring=torch.flatten(test)*self.k
                spring=spring.unsqueeze(1)
                fb=fb.unsqueeze(1)
                inertia=inertia.unsqueeze(1)
                AFB=fb.data.cpu().detach().numpy()
                Ain = inertia.data.cpu().detach().numpy()
                Aspring = spring.data.cpu().detach().numpy()
                AAAoutputs=combined.data.cpu().detach().numpy()
                AAAtimes=times.data.cpu().detach().numpy()
                AAAv=v_ext.data.cpu().detach().numpy()
                AAAa=a_ext.data.cpu().detach().numpy()
                AAAa2 = a_ext2.data.cpu().detach().numpy()
                c=inertia+spring-fb
                AAAc=c.data.cpu().detach().numpy()
                z=0
            with torch.autograd.no_grad():
                if epoch % 100 == 0:
                    print(epoch, "loss ", total_energy.data, "V", v_loss.data, "A",
                          a_loss.data, "Constituative", constitutative.data)
                    # print(epoch, "Traning Loss:", stress_divergence_loss.data, linear_elasticity_loss.data, volume_loss.data)
                energy[epoch] = total_energy.data



        times = torch.linspace(0, end, 101, requires_grad=True)
        x_values= torch.ones_like(times)*self.l #torch.linspace(0, self.l, 101, requires_grad=True)
        times = torch.reshape(times, (101, 1)).to(device)
        x_values = torch.reshape(x_values, (101, 1)).to(device)
        outputs = PINN(times,x_values)
        p = outputs[:, 0] * torch.flatten(times)* torch.flatten(x_values)
        v = outputs[:, 1] * torch.flatten(times)* torch.flatten(x_values)
        # p=torch.flatten(f / k - f / k * torch.cos(torch.sqrt(k / m) * torch.flatten(times)))
        # v_ext = torch.autograd.grad(p, times, torch.ones_like(p), create_graph=True, retain_graph=True)[0]
        # a_ext = torch.autograd.grad(v_ext, times, torch.ones_like(v_ext), create_graph=True, retain_graph=True)[0]
        v = outputs[:, 1] * torch.flatten(times)
        a = outputs[:, 2]
        plt.plot(times.data.cpu().detach().numpy(), 1 + p.data.cpu().detach().numpy(), label="PINN_started")
        #plt.show()

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # new_x=torch.linspace(0, end, 101, requires_grad=True)
        # new_x = torch.reshape(new_x, (101, 1))
        # newx_np=new_x.data.cpu().detach().numpy()
        # for i in times:
        #     time=torch.ones_like(new_x)*i
        #     outputs=PINN(time,new_x)
        #     p = outputs[:, 0] * torch.flatten(time) * torch.flatten(new_x)
        #     ax.scatter(time.data.cpu().detach().numpy(),newx_np,p.data.cpu().detach().numpy())
        #
        #     ax.scatter(time.data.cpu().detach().numpy(),newx_np,newx_np*(self.f/self.k-self.f/self.k*np.cos(np.sqrt(self.k/self.m)*time.data.cpu().detach().numpy())),color="pink")
        # ax.set_xlabel('time')
        # ax.set_ylabel('x')
        # ax.set_zlabel('disp')
        # plt.show()

        # plt.plot(times.data.cpu().detach().numpy(), 1 + v.data.cpu().detach().numpy(), label="PINN_v")
        # plt.plot(times.data.cpu().detach().numpy(), 1 + v_ext.data.cpu().detach().numpy(), label="PINN_v2")
        # plt.plot(times.data.cpu().detach().numpy(), 1 + a.data.cpu().detach().numpy(), label="PINN_a")
        # plt.plot(times.data.cpu().detach().numpy(), 1 + a_ext.data.cpu().detach().numpy(), label="PINN_a2")
        x0=p[-1]
        v0=v[-1]
        return PINN
    def PINN_continiuation(self,nets,prior_start,start,end):
        PINN = PINN_Net().to(device)
        PINN_optimizer = optimizer = torch.optim.Rprop(PINN.parameters(), lr=.0002, step_sizes=[1e-8, 1])
        iterations = 1000
        num_times = 1000
        energy = np.zeros(iterations)
        loss_function = torch.nn.MSELoss()

        inc=end-start

        for epoch in range(iterations):
            PINN_optimizer.zero_grad()
            start = start
            end = end
            times = torch.rand((num_times, 1), requires_grad=True).to(device) * (inc)
            x_values = torch.rand((num_times, 1), requires_grad=True).to(device) * self.l * 1.1
            if epoch == 999:
                x_values = torch.linspace(0, self.l, num_times, requires_grad=True)
                x_values = torch.reshape(x_values, (num_times, 1)).to(device)
                times = torch.linspace(0, inc, num_times, requires_grad=True)
                times = torch.reshape(times, (num_times, 1)).to(device)

            f = torch.tensor(self.f)
            k = torch.tensor(self.k)
            m = torch.tensor(self.m)

            outputs = PINN(times, x_values)
            old_ouput=torch.zeros_like(outputs)
            old_times=torch.ones_like(times)*inc
            # old_ouput=old_net(old_times,x_values)
            for net in nets:
                old_ouput+=net(old_times,x_values)

            #################### updating the boundary conditions is the key to the "continuation" ####################
            p = outputs[:, 0] *torch.flatten(times)* torch.flatten(x_values)  +old_ouput[:,0]* inc* torch.flatten(x_values)
            # p = torch.flatten(f/k-f/k*torch.cos(torch.sqrt(k/m)*times))            #Loss is lower (this is the ground truth)
            v = outputs[:, 1] *torch.flatten(times)* torch.flatten(x_values)  +old_ouput[:,1]*inc* torch.flatten(x_values)
            a = outputs[:, 2] *torch.flatten(x_values)                        +old_ouput[:,2]* torch.flatten(x_values)
            s = outputs[:, 3]

            combined = torch.stack((p, v, a, s), dim=1)

            s_ext = p

            strain = torch.autograd.grad(p, x_values, torch.ones_like(p), create_graph=True, retain_graph=True)[0]

            #Strain vs test is the major issue here!!!!!!!!!!!!
            test=torch.flatten(p)/torch.flatten(x_values)

            v_ext = torch.autograd.grad(p, times, torch.ones_like(p), create_graph=True, retain_graph=True)[0]
            a_ext2 = torch.autograd.grad(v_ext, times, torch.ones_like(times), create_graph=True, retain_graph=True)[0]
            a_ext = torch.autograd.grad(v, times, torch.ones_like(p), create_graph=True, retain_graph=True)[0]

            v_loss = loss_function(v, torch.flatten(v_ext)) * 10
            a_loss = loss_function(a, torch.flatten(a_ext)) * 1

            fb = self.f * torch.ones_like(p)
            stress_zeros = torch.zeros_like(fb)
            # AAAA=torch.flatten(a_ext2) * self.m
            # BBBB=p * self.k
            actual = torch.flatten(a) * self.m + torch.flatten(test) * self.k - fb*torch.flatten(x_values)

            constitutative = loss_function(actual, stress_zeros) * 1

            weighting = torch.tensor([1, epoch / iterations])
            total_energy = constitutative + v_loss + a_loss

            # we need to add some term related to the elastic force working on the system

            total_energy.backward()
            PINN_optimizer.step()
            if epoch == 999:
                inertia = torch.flatten(a_ext2) * self.m
                spring = p * self.k
                spring = spring.unsqueeze(1)
                fb = fb.unsqueeze(1)
                inertia = inertia.unsqueeze(1)
                AFB = fb.data.cpu().detach().numpy()
                Ain = inertia.data.cpu().detach().numpy()
                Aspring = spring.data.cpu().detach().numpy()
                AAAoutputs = combined.data.cpu().detach().numpy()
                AAAtimes = times.data.cpu().detach().numpy()
                AAAv = v_ext.data.cpu().detach().numpy()
                AAAa = a_ext.data.cpu().detach().numpy()
                AAAa2 = a_ext2.data.cpu().detach().numpy()
                c = inertia + spring - fb
                AAAc = c.data.cpu().detach().numpy()
                z = 0
            with torch.autograd.no_grad():
                if epoch % 100 == 0:
                    print(epoch, "loss ", total_energy.data, "V", v_loss.data, "A",
                          a_loss.data, "Constituative", constitutative.data)
                    # print(epoch, "Traning Loss:", stress_divergence_loss.data, linear_elasticity_loss.data, volume_loss.data)
                energy[epoch] = total_energy.data

        times = torch.linspace(0, inc, 101, requires_grad=True)
        x_values = torch.ones_like(times) * self.l  # torch.linspace(0, self.l, 101, requires_grad=True)
        times = torch.reshape(times, (101, 1)).to(device)
        x_values = torch.reshape(x_values, (101, 1)).to(device)
        outputs = PINN(times, x_values)
        old_times = torch.ones_like(times) * (inc)
        old_ouput=torch.zeros_like(outputs)
        for net in nets:
            old_ouput += net(old_times, x_values)
        p = outputs[:, 0]  *torch.flatten(times)* torch.flatten(x_values)  +old_ouput[:,0]* inc* torch.flatten(x_values)
        # v = outputs[:, 1] *torch.flatten(times)* torch.flatten(x_values)  +old_ouput[:,1]* (start-prior_start)* torch.flatten(x_values)
        # # p=torch.flatten(f / k - f / k * torch.cos(torch.sqrt(k / m) * torch.flatten(times)))
        # v_ext = torch.autograd.grad(p, times, torch.ones_like(p), create_graph=True, retain_graph=True)[0]
        # a_ext = torch.autograd.grad(v_ext, times, torch.ones_like(v_ext), create_graph=True, retain_graph=True)[0]
        # a = outputs[:, 2]
        plt.plot(start+times.data.cpu().detach().numpy(), 1 + p.data.cpu().detach().numpy(), label="PINN")
        # plt.plot(times.data.cpu().detach().numpy(), 1 + v.data.cpu().detach().numpy(), label="PINN_v")
        # plt.plot(times.data.cpu().detach().numpy(), 1 + v_ext.data.cpu().detach().numpy(), label="PINN_v2")
        # plt.plot(times.data.cpu().detach().numpy(), 1 + a.data.cpu().detach().numpy(), label="PINN_a")
        # plt.plot(times.data.cpu().detach().numpy(), 1 + a_ext.data.cpu().detach().numpy(), label="PINN_a2")
        return PINN
    def show_plot(self):
        plt.legend()
        plt.show()

m=.1
k=8
f=1
l=1
comparison=method_comparision(m,k,l,f)
comparison.analytic()
a=.25
nets=[]
t0=time.time()
net0=comparison.PINN(a)
nets.append(net0)
for i in range(101D):
    print("Starting Sim Time: %5.2F"% ((i+1)*a))
    output=comparison.PINN_continiuation(nets,i*a,(i+1)*a,(i+2)*a)
    nets.append(output)
t1=time.time()
print("Elapsed Compute Time: %5.2F"%(t1-t0))
# nets.append(net2)
# net3=comparison.PINN_continiuation(nets,.4,.6,.8)

#comparison.DEM()
comparison.show_plot()

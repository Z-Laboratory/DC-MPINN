import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PINN_Net(nn.Module):
    def __init__(self):
        super(PINN_Net, self).__init__()
        a = 50 #increasing this is a great way to inprove accuracy (at the cost of compute time)
        self.hidden_layer1 = nn.Linear(2, a)
        self.hidden_layer2 = nn.Linear(a, a)
        self.hidden_layer3 = nn.Linear(a, a)
        self.hidden_layer4 = nn.Linear(a, a)
        self.hidden_layer5 = nn.Linear(a, a)
        self.hidden_layer6 = nn.Linear(a, a)
        self.hidden_layer7 = nn.Linear(a, a)
        self.hidden_layer8 = nn.Linear(a, a)
        self.output_layer = nn.Linear(a, 4)
        #self.activation = torch.relu
        #self.activation=torch.nn.LeakyReLU()
        self.activation=torch.tanh
    def forward(self,t,x):
        if t.dim()==1:
            t=torch.unsqueeze(t,1)
        if x.dim()==1:
            x=torch.unsqueeze(x,1)
        inputs = torch.cat([t,x], axis=1)  # combined two arrays of 1 columns each to one array of 2 columns
        layer1_out = self.activation(self.hidden_layer1(inputs))
        layer2_out = self.activation(self.hidden_layer2(layer1_out))
        layer3_out = self.activation(self.hidden_layer3(layer2_out))
        layer4_out = self.activation(self.hidden_layer4(layer3_out))
        layer5_out = self.activation(self.hidden_layer5(layer4_out))
        layer6_out = self.activation(self.hidden_layer6(layer5_out))+layer3_out
        # layer7_out = self.activation(self.hidden_layer7(layer6_out))
        # layer8_out = self.activation(self.hidden_layer8(layer7_out))
        output = self.output_layer(layer3_out)
        return output

class AUX_Net(nn.Module):
    def __init__(self):
        super(AUX_Net, self).__init__()
        a = 25 #increasing this is a great way to inprove accuracy (at the cost of compute time)
        self.hidden_layer1 = nn.Linear(2, a)
        self.hidden_layer2 = nn.Linear(a, a)
        self.hidden_layer3 = nn.Linear(a, a)
        self.hidden_layer4 = nn.Linear(a, a)
        self.hidden_layer5 = nn.Linear(a, a)
        self.hidden_layer6 = nn.Linear(a, a)
        self.hidden_layer7 = nn.Linear(a, a)
        self.hidden_layer8 = nn.Linear(a, a)
        self.output_layer = nn.Linear(a, 4)
        #self.activation = torch.relu
        #self.activation=torch.nn.LeakyReLU()
        self.activation=torch.tanh
    def forward(self,t,x):
        if t.dim()==1:
            t=torch.unsqueeze(t,1)
        if x.dim()==1:
            x=torch.unsqueeze(x,1)
        inputs = torch.cat([t,x], axis=1)  # combined two arrays of 1 columns each to one array of 2 columns
        layer1_out = self.activation(self.hidden_layer1(inputs))
        layer2_out = self.activation(self.hidden_layer2(layer1_out))
        layer3_out = self.activation(self.hidden_layer3(layer2_out))
        # layer4_out = self.activation(self.hidden_layer4(layer3_out))
        # layer5_out = self.activation(self.hidden_layer5(layer4_out))
        # layer6_out = self.activation(self.hidden_layer6(layer5_out))+layer3_out
        # layer7_out = self.activation(self.hidden_layer7(layer6_out))
        # layer8_out = self.activation(self.hidden_layer8(layer7_out))
        output = self.output_layer(layer3_out)
        return output

class PINN():
    def __init__(self,m,k,l,f,end):
        self.mass=m
        self.k=k
        self.f=f
        self.l=l
        self.m=m
        self.end=end
        self.loss_function = torch.nn.MSELoss()
    def run(self):

        PINN = PINN_Net().to(device)
        PINN_optimizer=torch.optim.Rprop(PINN.parameters(), lr=.02, step_sizes=[1e-16, 10])
        iterations = 1
        num_times = 20000
        losses = np.zeros(iterations)
        offset = 1000

        for epoch in range(iterations):
            PINN_optimizer.zero_grad()
            times_start = torch.ones((offset, 1), requires_grad=True).to(device) * 0
            times_mid = torch.rand((num_times-offset*2, 1), requires_grad=True).to(device)*self.end
            times_end = torch.ones((offset,1), requires_grad=True).to(device)*self.end
            times=torch.cat((times_mid,times_start,times_end))

            x_values_mid=torch.rand((num_times-offset*2, 1), requires_grad=True).to(device)
            x_values0 = torch.ones((offset, 1), requires_grad=True).to(device) * self.l
            x_values1 = torch.ones((offset, 1), requires_grad=True).to(device) *0
            x_values = torch.cat((x_values0, x_values1,x_values_mid))

            p,v,a,s,m=self.get_ouptuts(times,x_values,PINN)

            total_loss,v_loss,a_loss,stress_loss,momentum_loss=self.get_loss(times,x_values,p,v,a,s,m)

            total_loss.backward()
            PINN_optimizer.step()

            with torch.autograd.no_grad():
                if epoch % 100 == 0:
                    print(epoch, "loss ", total_loss.data,"V",v_loss.data,"A",a_loss.data,"Stress",stress_loss.data,"momentum",momentum_loss.data)
                losses[epoch] = total_loss.data


        plt.plot(losses)
        plt.yscale('log')
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x_values.data.cpu().detach().numpy(),times.data.cpu().detach().numpy(),p.data.cpu().detach().numpy())
        ax.set_xlabel('X ')
        ax.set_ylabel('Time')
        ax.set_zlabel('Position')
        plt.show()

        mask = torch.ceil(torch.max(times + x_values - 1, torch.zeros_like(times)))
        g = torch.flatten((x_values + times - 1 - (torch.sin(torch.pi * (x_values + times - 1))) / torch.pi) * mask)-torch.flatten(p)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x_values.data.cpu().detach().numpy(),times.data.cpu().detach().numpy(),g.data.cpu().detach().numpy())
        ax.set_xlabel('X ')
        ax.set_ylabel('Time')
        ax.set_zlabel('Position')
        plt.show()

        return PINN
    def get_ouptuts(self, times, x_values, PINN):
        outputs = PINN(times, x_values)
        p = outputs[:, 0] * torch.flatten(x_values) * torch.flatten(times)+torch.flatten(x_values)
        v = outputs[:, 1] * torch.flatten(x_values) * torch.flatten(times)
        a = outputs[:, 2] * torch.flatten(x_values)
        s = outputs[:, 3]
        return p, v, a, s
    def get_aux_ouptuts(self, times, x_values, PINN):
        outputs = PINN(times, x_values)
        p = outputs[:, 0] * torch.flatten(x_values)
        v = outputs[:, 1] #* torch.flatten(x_values)
        a = outputs[:, 2] #* torch.flatten(x_values)
        s = outputs[:, 3]
        return p, v, a, s
    def get_continuation_ouptuts(self,times,x_values,PINN,AUX):
        outputs = PINN(times, x_values)
        aux_times=torch.zeros_like(times)
        aux_p,aux_v,aux_a,aux_s=self.get_aux_ouptuts(aux_times,x_values,AUX)
        aux_p.grad=None
        aux_v.grad=None
        aux_a.grad=None
        aux_s.grad=None
        p = outputs[:, 0] * torch.flatten(x_values)*torch.flatten(times)+aux_p
        v = outputs[:, 1] * torch.flatten(x_values)*torch.flatten(times)+aux_v
        a = outputs[:, 2] * torch.flatten(x_values)*torch.flatten(times)+aux_a
        s = outputs[:, 3]                          *torch.flatten(times)+aux_s
        return p, v, a, s
    def get_loss(self, times, x_values, p, v, a, s):
        strain = torch.autograd.grad(p, x_values, torch.ones_like(p), create_graph=True, retain_graph=True)[0]
        v_ext = torch.autograd.grad(p, times, torch.ones_like(p), create_graph=True, retain_graph=True)[0]
        a_ext = torch.autograd.grad(v, times, torch.ones_like(v), create_graph=True, retain_graph=True)[0]
        stress_gradient = torch.autograd.grad(s, x_values, torch.ones_like(s), create_graph=True, retain_graph=True)[0]

        stress_DNN = self.k * torch.flatten(strain)

        # Do I need to apply conservation of momentum here??
        conservation_of_momentum = (m-(-torch.flatten(stress_gradient) + self.m * torch.flatten(a)))

        stress_loss = self.loss_function(stress_DNN, s) * 1
        momentum_loss = self.loss_function(conservation_of_momentum, torch.zeros_like(conservation_of_momentum)) * 1
        v_loss = self.loss_function(v, torch.flatten(v_ext)) * 1
        a_loss = self.loss_function(a, torch.flatten(a_ext)) * 1

        total_energy = momentum_loss + v_loss + a_loss + stress_loss
        return total_energy, v_loss, a_loss, stress_loss, momentum_loss
    def plot(self,PINN,start,duration,aux):
        num_points=11
        num_times=200
        x=torch.linspace(0,self.l,num_points,requires_grad=True).to(device)
        t=torch.linspace(0, duration, num_times, requires_grad=True).to(device)

        gridx,gridt=torch.meshgrid(x,t,indexing='ij')
        gridx=torch.flatten(gridx)
        gridt=torch.flatten(gridt)
        if start==0:
            p,v,a,s=self.get_ouptuts(gridt,gridx,PINN)
        elif aux==True:
            p, v, a, s = self.get_aux_ouptuts(gridt, gridx, PINN)
        else:
            p,v,a,s=self.get_continuation_ouptuts(gridt, gridx, PINN,aux)

        p_pt = torch.reshape(p, (num_points, num_times))
        v = torch.reshape(v, (num_points, num_times))
        a = torch.reshape(a, (num_points, num_times))
        s = torch.reshape(s, (num_points, num_times))

        p = np.array(p_pt.tolist())
        v = np.array(v.tolist())
        a = np.array(a.tolist())
        s = np.array(s.tolist())

        ax.scatter(gridx.data.cpu().detach().numpy(),gridt.data.cpu().detach().numpy()+start,p_pt.data.cpu().detach().numpy())
        ax.set_xlabel('X ')
        ax.set_ylabel('Time')
        ax.set_zlabel('Position')
    def get_power(self,PINN,end):
        num_points=11
        num_times=200
        x=torch.linspace(0,self.l,num_points,requires_grad=True).to(device)
        t=torch.linspace(0, end, num_times, requires_grad=True).to(device)
        gridx,gridt=torch.meshgrid(x,t,indexing='ij')
        gridx=torch.flatten(gridx)
        gridt=torch.flatten(gridt)
        p,v,a,s=self.get_ouptuts(gridt,gridx,PINN)
        p = torch.reshape(p, (num_points, num_times))
        v = torch.reshape(v, (num_points, num_times))
        a = torch.reshape(a, (num_points, num_times))
        s = torch.reshape(s, (num_points, num_times))

        #3 sources of power
        #boundary power =F_ext*v[-1]
        #spring power =.5*s*v
        #kinetic power= .5*m*v*a

        boundary_p=v[-1,:]*self.f
        spring_p=torch.mean(s*v,0)#Should there be a factor of 1/2 here??
        kinetic_p=torch.mean(self.m*v*a,0)
        total_energy=spring_p+kinetic_p-boundary_p
        total_power=total_energy.tolist()
        plt.plot(t.tolist(),total_power,label="total power")
        plt.plot(t.tolist(), spring_p.tolist(),label="spring power")
        plt.plot(t.tolist(), kinetic_p.tolist(),label="kinetic power")
        plt.plot(t.tolist(), boundary_p.tolist(),label="boundary power")
        plt.legend()
        plt.show()
    def get_energy(self,PINN,start,duration,aux):
        num_points = 10
        num_times=50
        x = torch.linspace(0, self.l, num_points, requires_grad=True).to(device)
        t = torch.linspace(0, duration, num_times, requires_grad=True).to(device)
        gridx, gridt = torch.meshgrid(x, t, indexing='ij')
        gridx = torch.flatten(gridx)
        gridt = torch.flatten(gridt)
        if start == 0:
            p, v, a, s = self.get_ouptuts(gridt, gridx, PINN)
        elif aux == True:
            p, v, a, s = self.get_aux_ouptuts(gridt, gridx, PINN)
        else:
            p, v, a, s = self.get_continuation_ouptuts(gridt, gridx, PINN, aux)
        strain = torch.autograd.grad(p, gridx, torch.ones_like(p), create_graph=True, retain_graph=True)[0]

        stress_grad = torch.autograd.grad(s, gridx, torch.ones_like(s), create_graph=True, retain_graph=True)[0]
        p = torch.reshape(p, (num_points, num_times))
        v = torch.reshape(v, (num_points, num_times))
        a = torch.reshape(a, (num_points, num_times))
        s = torch.reshape(s, (num_points, num_times))
        strain = torch.reshape(strain, (num_points, num_times))
        stress_grad = torch.reshape(stress_grad, (num_points, num_times))
        #
        # AAp=p.data.cpu().detach().numpy()
        # AAv=v.data.cpu().detach().numpy()
        # AAA=a.data.cpu().detach().numpy()
        # AAs=s.data.cpu().detach().numpy()
        # AAsgrad=stress_grad.data.cpu().detach().numpy()
        # AAstrain = strain.data.cpu().detach().numpy()
        # quit()

        #energy has 2 terms (since boundary energy is 0)
        #kinetic energy = .5*m*v^2
        #potential energy = .5*k*x^2
        kinetic_e = torch.mean(2*self.m * v * v, 0)
        spring_e = torch.mean(.5 * s * strain, 0)
        #spring_e = torch.mean(.5 * s * p, 0)
        total_energy = spring_e + kinetic_e
        total_energy = total_energy.tolist()
        t_revised=np.array(t.tolist())+start
        plt.plot(t_revised, total_energy, label="total energy")
        plt.plot(t_revised, spring_e.tolist(), label="spring energy")
        plt.plot(t_revised, kinetic_e.tolist(), label="kinetic energy")
        plt.legend()


m=1
k=2
l=1
f=1*0
duration=.25
PINN=PINN(m,k,l,f,duration)
run=False
multi_segment=True

if run:
    mpin0=PINN.run()
    torch.save(mpin0.state_dict(), "mpin0.pt")
if multi_segment:
    # mpinn075 = PINN_Net()
    # mpinn075.load_state_dict(torch.load("mpinn075.pt", map_location=torch.device('cpu')))
    # PINN.get_energy(mpinn075, start=.0, duration=.75, aux=False)
    num_models=3
    end_time=9
    interval = end_time / num_models
    mpinn = PINN_Net()
    maux = AUX_Net()
    mpinn.load_state_dict(torch.load("start_model.pt", map_location=torch.device('cpu')))
    maux.load_state_dict(torch.load("start_aux.pt", map_location=torch.device('cpu')))
    PINN.get_energy(mpinn, start=0, duration=interval, aux=False)

    for i in range(num_models-1):
        mpinn=PINN_Net()
        mpinn.load_state_dict(torch.load("model"+str(i+1)+".pt", map_location=torch.device('cpu')))
        PINN.get_energy(mpinn, start=(i+1)*interval, duration=interval, aux=maux)
        maux=AUX_Net()
        maux.load_state_dict(torch.load("aux"+str(i+1)+".pt", map_location=torch.device('cpu')))
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    mpinn = PINN_Net()
    maux = AUX_Net()
    mpinn.load_state_dict(torch.load("start_model.pt", map_location=torch.device('cpu')))
    maux.load_state_dict(torch.load("start_aux.pt", map_location=torch.device('cpu')))
    PINN.plot(mpinn, start=0, duration=interval, aux=False)
    for i in range(num_models-1):
        mpinn=PINN_Net()
        mpinn.load_state_dict(torch.load("model"+str(i+1)+".pt", map_location=torch.device('cpu')))
        PINN.plot(mpinn, start=(i+1)*interval, duration=interval, aux=maux)
        maux = AUX_Net()
        maux.load_state_dict(torch.load("aux" + str(i + 1) + ".pt", map_location=torch.device('cpu')))

    plt.show()
    quit()
else:
    mpinn=PINN_Net()
    maux=AUX_Net()
    maux1=AUX_Net()
    mpinn1 = PINN_Net()
    mpinn2 = PINN_Net()
    mpinn05 = PINN_Net()
    mpinn075 = PINN_Net()
    mpinn40 = PINN_Net()
    mpinn075.load_state_dict(torch.load("mpinn40.pt", map_location=torch.device('cpu')))
    mpinn075.load_state_dict(torch.load("mpinn075.pt", map_location=torch.device('cpu')))
    mpinn.load_state_dict(torch.load("mpinn0.pt",map_location=torch.device('cpu')))
    maux.load_state_dict(torch.load("mAux0.pt",map_location=torch.device('cpu')))
    mpinn1.load_state_dict(torch.load("mpinn1.pt", map_location=torch.device('cpu')))
    maux1.load_state_dict(torch.load("mAux1.pt", map_location=torch.device('cpu')))
    mpinn2.load_state_dict(torch.load("mpinn2.pt", map_location=torch.device('cpu')))

    # mpinn.load_state_dict(torch.load("start_model.pt",map_location=torch.device('cpu')))
    # maux.load_state_dict(torch.load("start_aux.pt",map_location=torch.device('cpu')))
    # mpinn1.load_state_dict(torch.load("model1.pt", map_location=torch.device('cpu')))

    mpinn05.load_state_dict(torch.load("mpinn0_5.pt", map_location=torch.device('cpu')))


    # PINN.get_power(mpinn, end)
    PINN.get_energy(mpinn, start=0,duration=.25,aux=False)
    PINN.get_energy(mpinn40, start=0, duration=40, aux=False)
    PINN.get_energy(mpinn1, start=.25, duration=.25,aux=maux)
    PINN.get_energy(mpinn1, start=.5, duration=.25, aux=maux1)
    #PINN.get_energy(mpinn05, start=.0, duration=.5,aux=False)
    PINN.get_energy(mpinn075, start=.0, duration=.75, aux=False)
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    PINN.plot(mpinn40, start=0, duration=40, aux=False)
    PINN.plot(mpinn,start=0,duration=duration,aux=False)
    PINN.plot(maux,start=.25,duration=0,aux=True)
    PINN.plot(mpinn1, start=.25, duration=.25, aux=maux)
    PINN.plot(mpinn2, start=.5, duration=.25, aux=maux1)
    PINN.plot(maux1, start=.5, duration=0, aux=True)
    PINN.plot(mpinn075, start=.0, duration=.75, aux=False)
    #PINN.plot(mpinn05, start=0, duration=.5, aux=False)

    # mpinn=PINN_Net()
    # mpinn.load_state_dict(torch.load("mpin_1s.pt",map_location=torch.device('cpu')))
    # PINN.plot(mpinn,1)
    # mpinn2=PINN_Net()
    # mpinn2.load_state_dict(torch.load("mpin_2s.pt",map_location=torch.device('cpu')))
    # PINN.plot(mpinn2,2)
    # mpinn3=PINN_Net()
    # mpinn3.load_state_dict(torch.load("mpin_3s.pt",map_location=torch.device('cpu')))
    # PINN.plot(mpinn3,3)
    plt.show()
    x=0
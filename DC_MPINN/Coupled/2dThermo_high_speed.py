import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

class PINN_Net(nn.Module):
    def __init__(self):
        super(PINN_Net, self).__init__()
        a = 250 #increasing this is a great way to inprove accuracy (at the cost of compute time)
        #good option is a=500, lr=.004 iterations=501
        self.hidden_layer1 = nn.Linear(3, a)
        self.hidden_layer2 = nn.Linear(a, a)
        self.hidden_layer3 = nn.Linear(a, a)
        self.hidden_layer4 = nn.Linear(a, a)
        self.hidden_layer5 = nn.Linear(a, a)
        self.hidden_layer6 = nn.Linear(a, a)
        self.hidden_layer7 = nn.Linear(a, a)
        self.hidden_layer8 = nn.Linear(a, a)
        self.output_layer = nn.Linear(a, 3)
        #self.activation = torch.relu
        #self.activation=torch.nn.LeakyReLU()
        self.activation=torch.tanh
    def forward(self,t,x,y):
        inputs = torch.cat([t,x,y], axis=1)  # combined two arrays of 1 columns each to one array of 2 columns
        layer1_out = self.activation(self.hidden_layer1(inputs))
        layer2_out = self.activation(self.hidden_layer2(layer1_out))
        layer3_out = self.activation(self.hidden_layer3(layer2_out))
        layer4_out = self.activation(self.hidden_layer4(layer3_out))+layer2_out
        layer5_out = self.activation(self.hidden_layer5(layer4_out))
        layer6_out = self.activation(self.hidden_layer6(layer5_out))+layer3_out
        layer7_out = self.activation(self.hidden_layer7(layer6_out))
        layer8_out = self.activation(self.hidden_layer8(layer7_out))+layer5_out
        output = self.output_layer(layer4_out)
        return output

class Thermo_PINN():
    def __init__(self,thermal_conductivity,specific_heat_capacity,density):
        self.k=thermal_conductivity
        self.c=specific_heat_capacity
        self.density=density
        self.loss_function = torch.nn.MSELoss()
        self.offset = 500 # basically the number of points on each boundary (x=0,x=1,t=0,t=1)
        self.num_times=5000 #number of total points in simulation
        self.num_aux=2000
        self.l=1
    def run(self,end):
        PINN = PINN_Net().to(device)
        PINN_optimizer=torch.optim.Rprop(PINN.parameters(), lr=.01,etas=[.8,1.1], step_sizes=[1e-16, 100])
        #PINN_optimizer=torch.optim.SGD(PINN.parameters(),lr=.00002)
        #PINN_optimizer=torch.optim.Adam(PINN.parameters(),lr=.0002,weight_decay=.001)
        iterations = 301
        losses = np.zeros(iterations)
        end = end

        #option: generate random points once. This reduces compute time by about 10% however, each epoch is no longer independent
        #however if the batch size is sufficiently large this may not matter
        #we can also (potentially) reduce the number of iterations
        times_end = torch.ones((self.offset,1), requires_grad=False).to(device)*end
        #non_uniform time slicing
        times_mid1 = torch.rand((self.num_times//2-self.offset, 1), requires_grad=False).to(device)*end/5
        times_mid2 = torch.rand((self.num_times//2-self.offset, 1), requires_grad=False).to(device)*4*end/5+end/5
        times_mid=torch.cat((times_mid1,times_mid2))
        x_values_mid=torch.rand((self.num_times-self.offset*2, 1), requires_grad=False).to(device)
        x_values1 = torch.ones((self.offset, 1), requires_grad=False).to(device) * self.l
        y_values_mid=torch.rand(((self.num_times-self.offset*2)//2, 1), requires_grad=False).to(device)
        y_values1 = torch.ones((self.offset, 1), requires_grad=False).to(device) * self.l

        for epoch in range(iterations):
            PINN_optimizer.zero_grad()

            
            times_start = torch.zeros((self.offset, 1), requires_grad=True).to(device) * 0
            times=torch.cat((times_mid,times_end,times_start))


            x_values0 = torch.zeros((self.offset, 1), requires_grad=True).to(device) *0
            x_values = torch.cat((x_values0, x_values1,x_values_mid))

            y_values0 = torch.zeros((self.offset, 1), requires_grad=True).to(device) *0
            y_values=torch.cat((y_values_mid,y_values0, y_values1,y_values_mid))

            sumed_values=times*.1+x_values*1+y_values*1 #default it .1,1,1

            self.mask=torch.gt(sumed_values,torch.ones_like(sumed_values)*.025) #default is compared to .25
            times=times*self.mask
            x_values=x_values*self.mask

            outputs=self.get_ouptuts(times,x_values,y_values,PINN)
            total_loss,heat_equation_loss,continuity_loss=self.get_loss(times,x_values,y_values,outputs)


            total_loss.backward()
            PINN_optimizer.step()
            with torch.autograd.no_grad():
                if epoch % 100 == 0:
                    print(epoch, "loss ", total_loss.data,"heat loss", heat_equation_loss.data,"continuity_loss",continuity_loss.data)#,"BC loss",other_BC_loss.data)#,"V",v_loss.data,"A",a_loss.data,"Stress",stress_loss.data,"boundary",boundary_loss.data,"wave",wave_loss,"energy",energy)
                    #values_np=values.data.cpu().detach().numpy()
                    #np.savetxt("values.csv",values_np)

                losses[epoch] = total_loss.data

        plt.plot(losses)
        plt.yscale('log')
        plt.show()
        plt.savefig("first_loss.png")
        return PINN
    def get_ouptuts(self, times, x_values,y_values, PINN):
        outputs = PINN(times, x_values,y_values)

        t=outputs[:,0]
        dtdx=torch.flatten(outputs[:,1])*torch.flatten(self.l-x_values)
        dtdy=torch.flatten(outputs[:,2])*torch.flatten(self.l-y_values)

        #basic 2d 
        #modified_ouputs=torch.flatten(t)*torch.flatten(torch.minimum(times, torch.ones_like(times)))*torch.flatten(x_values)+torch.minimum(25*torch.flatten(times),25*torch.ones_like(torch.flatten(times)))*torch.flatten(self.l-x_values)

        #More complex 2d
        modified_ouputs=torch.flatten(t)*torch.flatten(torch.minimum(times, torch.ones_like(times)))*torch.flatten(x_values)*torch.flatten(y_values)+torch.minimum(25*torch.flatten(times),25*torch.ones_like(torch.flatten(times)))
        t=modified_ouputs

        outputs=torch.stack((t,dtdx,dtdy),dim=1)
        return outputs
    def get_loss(self, times, x_values,y_values, outputs):
        t=outputs[:,0]
        t=torch.unsqueeze(t,1)
        dtdx=outputs[:,1]
        dtdx=torch.unsqueeze(dtdx,1)
        dtdy=outputs[:,2]
        dtdy=torch.unsqueeze(dtdy,1)

        auto_dTdt = torch.autograd.grad(t, times, torch.ones_like(t), create_graph=True, retain_graph=True)[0]

        temp_gradient_x = torch.autograd.grad(t, x_values, torch.ones_like(t), create_graph=True, retain_graph=True)[0]
        second_dertivative_x=torch.autograd.grad(dtdx, x_values, torch.ones_like(t), create_graph=True, retain_graph=True)[0]
        temp_gradient_y = torch.autograd.grad(t, y_values, torch.ones_like(t), create_graph=True, retain_graph=True)[0]
        second_dertivative_y=torch.autograd.grad(dtdy, y_values, torch.ones_like(t), create_graph=True, retain_graph=True)[0]

        #print(second_dertivative)
        #dt doesn't work here because is is a 
        heat_equation_values= (auto_dTdt*self.c*self.density-(self.k*second_dertivative_x+self.k*second_dertivative_y))*self.mask
        heat_equation_values_inner=heat_equation_values[1*self.offset:-1*self.offset]
        heat_equation_loss=self.loss_function(heat_equation_values_inner,torch.zeros_like(heat_equation_values_inner))

        continuity_loss=(self.loss_function(dtdx,temp_gradient_x)+self.loss_function(dtdy,temp_gradient_y))*10

        total_loss=heat_equation_loss+continuity_loss#+IC_loss+BC_loss

        return total_loss,heat_equation_loss,continuity_loss
    def get_AUX_loss(self, times, x_values, p, v, a, s):
        strain = torch.autograd.grad(p, x_values, torch.ones_like(p), create_graph=True, retain_graph=True)[0]
        v_ext = torch.autograd.grad(p, times, torch.ones_like(p), create_graph=True, retain_graph=True)[0]
        a_ext = torch.autograd.grad(v, times, torch.ones_like(v), create_graph=True, retain_graph=True)[0]
        stress_gradient = torch.autograd.grad(s, x_values, torch.ones_like(s), create_graph=True, retain_graph=True)[0]

        stress_DNN = self.k * torch.flatten(strain)
        boundary_force=(self.f+s[-1])+(self.m*torch.flatten(a)[-1]) #TODO look at signs here
        boundary_loss=self.loss_function(boundary_force,torch.zeros_like(boundary_force))*1 #TODO assess if this boundary loss is a good solution to our problem of different physics at the boundary
        kinetic=2*self.m*torch.flatten(v)*torch.flatten(v)#TODO: Where did the 1/2 go?
        #spring=.5*torch.flatten(s)*torch.flatten(p) #this is based on the flawed assumtion that spring energy is from a "spring" Better to use strain energy density
        spring=.5*torch.flatten(s)*torch.flatten(strain)
        total_energy=torch.mean(kinetic[:-1]+spring[:-1])
        #inital_energy=torch.tensor((.5*self.k*self.offset+1/4*self.k*(self.num_times-2*self.offset))/self.num_times).to(device)
        inital_energy=torch.tensor(.5*self.k).to(device)
        energy_loss=self.loss_function(total_energy,inital_energy)*1
        wave_equation=(-torch.flatten(stress_gradient) + self.m * torch.flatten(a))[:-1] #we slice the loss becasue the wave equation does not hold when there is an external force
        wave_loss=self.loss_function(wave_equation,torch.zeros_like(wave_equation))*1
        stress_loss = self.loss_function(stress_DNN, s) * 1
        v_loss = self.loss_function(v, torch.flatten(v_ext)) * 1
        a_loss = self.loss_function(a, torch.flatten(a_ext)) * 1
        total_loss = boundary_loss + v_loss+a_loss+stress_loss+wave_loss+energy_loss #we remove v and A loss. 
        return total_loss, v_loss, a_loss, stress_loss, boundary_loss, wave_loss,energy_loss

    # def plot(self,PINN):
    #     num_points=101
    #     x=torch.linspace(0,self.l,num_points,requires_grad=True).to(device)
    #     t=torch.linspace(0, 1, num_points, requires_grad=True).to(device)
    #     gridx,gridt=torch.meshgrid(x,t,indexing='ij')
    #     p,v,a,s,m=self.get_ouptuts(gridt,gridx,PINN)
    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection='3d')
    #     ax.scatter(gridx.data.cpu().detach().numpy(),gridt.data.cpu().detach().numpy(),p.data.cpu().detach().numpy())
    #     ax.set_xlabel('X ')
    #     ax.set_ylabel('Time')
    #     ax.set_zlabel('Position')
    #     plt.show()
    def full_run(self,num_models,end_time):
        interval=end_time/num_models

        models=[]
        aux=[]
        start_model=PINN.run(interval)
        models.append(start_model)
        torch.save(start_model.state_dict(), "2d_thermo_start_model.pt")
        start_aux=PINN.train_AUX(start_model,interval,old_aux=None)
        aux.append(start_aux)
        torch.save(start_aux.state_dict(), "2d_thermo_start_aux.pt")
        for i in range(num_models-1):
            mpinn=PINN.continuation(interval,aux[-1])
            mAux=PINN.train_AUX(mpinn,interval,old_aux=aux[-1])
            models.append(mpinn)
            aux.append(mAux)
            torch.save(mpinn.state_dict(), "2d_thermo_model"+str(i+1)+".pt")
            torch.save(mAux.state_dict(), "2d_thermo_aux"+str(i+1)+".pt")


specific_heat_capacity=1
thermal_conductivity=.1
density=1
PINN=Thermo_PINN(thermal_conductivity,specific_heat_capacity,density)
run=True
aux=False
mp1=False
aux2=False
mp2=False

t0=time.time()

#PINN.full_run(num_models=1,end_time=100)


if run:
    mpinn0=PINN.run(10)
    torch.save(mpinn0.state_dict(), "2d_thermo_mpinn0.pt")
if aux:
    mpinn0=PINN_Net().to(device)
    mpinn0.load_state_dict(torch.load("mpinn0.pt"))
    mAux0=PINN.train_AUX(mpinn0,.25,old_aux=None)
    torch.save(mAux0.state_dict(), "2d_thermo_mAux0.pt")
if mp1:
    mAux0=PINN_Net().to(device)
    mAux0.load_state_dict(torch.load("2d_thermo_mAux0.pt"))
    mpinn1=PINN.continuation(.25,mAux0)
    torch.save(mpinn1.state_dict(), "2d_thermo_mpinn1.pt")
if aux2:
    mAux0=PINN_Net().to(device)
    mpinn1=PINN_Net().to(device)
    mAux0.load_state_dict(torch.load("2d_thermo_mAux0.pt"))
    mpinn1.load_state_dict(torch.load("2d_thermo_mpinn1.pt"))
    mAux1=PINN.train_AUX(mpinn1,.25,old_aux=mAux0)
    torch.save(mAux1.state_dict(), "2d_thermo_mAux1.pt")
if mp2:
    mAux1=PINN_Net().to(device)
    mAux1.load_state_dict(torch.load("2d_thermo_mAux1.pt"))
    mpinn2=PINN.continuation(.25,mAux1)
    torch.save(mpinn2.state_dict(), "2d_thermo_mpinn2.pt")
t1=time.time()
print("Elapsed Compute Time: %5.2F"%(t1-t0))

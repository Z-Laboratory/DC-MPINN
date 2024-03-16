import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

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
        inputs = torch.cat([t,x], axis=1)  # combined two arrays of 1 columns each to one array of 2 columns
        layer1_out = self.activation(self.hidden_layer1(inputs))
        layer2_out = self.activation(self.hidden_layer2(layer1_out))
        layer3_out = self.activation(self.hidden_layer3(layer2_out))
        layer4_out = self.activation(self.hidden_layer4(layer3_out))#+layer2_out
        layer5_out = self.activation(self.hidden_layer5(layer4_out))
        layer6_out = self.activation(self.hidden_layer6(layer5_out))+layer3_out
        # layer7_out = self.activation(self.hidden_layer7(layer6_out))
        # layer8_out = self.activation(self.hidden_layer8(layer7_out))
        output = self.output_layer(layer3_out)
        return output

class Aux_Net(nn.Module):
    def __init__(self):
        super(Aux_Net, self).__init__()
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
        inputs = torch.cat([t,x], axis=1)  # combined two arrays of 1 columns each to one array of 2 columns
        layer1_out = self.activation(self.hidden_layer1(inputs))
        layer2_out = self.activation(self.hidden_layer2(layer1_out))
        layer3_out = self.activation(self.hidden_layer3(layer2_out))
        # layer4_out = self.activation(self.hidden_layer4(layer3_out))#+layer2_out
        # layer5_out = self.activation(self.hidden_layer5(layer4_out))
        # layer6_out = self.activation(self.hidden_layer6(layer5_out))+layer3_out
        output = self.output_layer(layer3_out)
        return output

class PINN():
    def __init__(self,m,k,l,f):
        self.mass=m
        self.k=k
        self.f=f
        self.l=l
        self.m=m
        self.loss_function = torch.nn.MSELoss()
        self.offset = 1000
        self.num_times=20000
        self.num_aux=2000
    def run(self,end):
        PINN = PINN_Net().to(device)
        PINN_optimizer=torch.optim.Rprop(PINN.parameters(), lr=.02, step_sizes=[1e-16, 10])
        iterations = 501
        losses = np.zeros(iterations)
        end = end

        for epoch in range(iterations):
            PINN_optimizer.zero_grad()
            times_start = torch.zeros((self.offset, 1), requires_grad=True).to(device) * 0
            times_mid = torch.rand((self.num_times-self.offset*2, 1), requires_grad=True).to(device)*end
            times_end = torch.ones((self.offset,1), requires_grad=True).to(device)*end
            times=torch.cat((times_mid,times_start,times_end))

            x_values_mid=torch.rand((self.num_times-self.offset*2, 1), requires_grad=True).to(device)
            x_values0 = torch.ones((self.offset, 1), requires_grad=True).to(device) * self.l
            x_values1 = torch.zeros((self.offset, 1), requires_grad=True).to(device) *0
            x_values = torch.cat((x_values0, x_values1,x_values_mid))

            p,v,a,s=self.get_ouptuts(times,x_values,PINN)
            total_loss,v_loss,a_loss,stress_loss,boundary_loss,wave_loss=self.get_loss(times,x_values,p,v,a,s)

            total_loss.backward()
            PINN_optimizer.step()
            with torch.autograd.no_grad():
                if epoch % 100 == 0:
                    print(epoch, "loss ", total_loss.data)#,"V",v_loss.data,"A",a_loss.data,"Stress",stress_loss.data,"boundary",boundary_loss.data,"wave",wave_loss,"energy",energy)
                losses[epoch] = total_loss.data
        return PINN
    def continuation(self,end,aux):
        CPinn = PINN_Net().to(device)
        CPinn_optimizer=torch.optim.Rprop(CPinn.parameters(), lr=.02, step_sizes=[1e-16, 10])
        iterations = 251
        losses = np.zeros(iterations)
        end = end

        for epoch in range(iterations):
            CPinn_optimizer.zero_grad()
            times_mid = torch.rand((self.num_times-self.offset, 1), requires_grad=True).to(device)*end
            times_end = torch.ones((self.offset,1), requires_grad=True).to(device)*end
            times=torch.cat((times_mid,times_end))

            x_values_mid=torch.rand((self.num_times-self.offset*2, 1), requires_grad=True).to(device)
            x_values0 = torch.ones((self.offset, 1), requires_grad=True).to(device) * self.l
            x_values1 = torch.zeros((self.offset, 1), requires_grad=True).to(device) *0
            x_values = torch.cat((x_values0, x_values1,x_values_mid))

            p,v,a,s=self.get_continuation_ouptuts(times,x_values,CPinn,aux)
            total_loss,v_loss,a_loss,stress_loss,boundary_loss,wave_loss=self.get_loss(times,x_values,p,v,a,s)

            total_loss.backward()
            CPinn_optimizer.step()
            with torch.autograd.no_grad():
                if epoch % 100 == 0:
                    print(epoch, "Continuation loss ", total_loss.data)#,"V",v_loss.data,"A",a_loss.data,"Stress",stress_loss.data,"boundary",boundary_loss.data,"wave",wave_loss,"energy",energy)
                losses[epoch] = total_loss.data
        return CPinn
    def train_AUX(self,PINN,end,old_aux):
        time_offset=0.0
        start=end-time_offset
        AUX = Aux_Net().to(device)
        AUX_optimizer=torch.optim.Rprop(AUX.parameters(), lr=.02, step_sizes=[1e-16, 10])
        iterations = 101
        losses = np.zeros(iterations)
        end = end
        for epoch in range(iterations):
            AUX_optimizer.zero_grad()
            # times_start = torch.zeros((self.offset, 1), requires_grad=True).to(device) +start
            # times_mid = torch.rand((self.num_times-self.offset*2, 1), requires_grad=True).to(device)*time_offset+start
            # times_end = torch.ones((self.offset,1), requires_grad=True).to(device)*end
            # times=torch.cat((times_mid,times_start,times_end))
            # x_values_mid=torch.rand((self.num_times-self.offset*2, 1), requires_grad=True).to(device)
            # x_values0 = torch.ones((self.offset, 1), requires_grad=True).to(device) * self.l
            # x_values1 = torch.ones((self.offset, 1), requires_grad=True).to(device) *0
            # x_values = torch.cat((x_values0, x_values1,x_values_mid))
            
            times=torch.ones((self.num_aux,1),requires_grad=True).to(device)*end
            aux_times=torch.zeros_like(times,requires_grad=True)
            x_values=torch.rand((self.num_aux,1),requires_grad=True).to(device)*self.l
            x_values[-1]=self.l
            if old_aux != None:
                p,v,a,s=self.get_continuation_ouptuts(times,x_values,PINN,old_aux)
            else:
                p,v,a,s=self.get_ouptuts(times,x_values,PINN)
            aux_p,aux_v,aux_a,aux_s=self.get_aux_ouptuts(aux_times,x_values,AUX)


            normal_loss,v_loss,a_loss,stress_loss,boundary_loss,wave_loss,energy=self.get_AUX_loss(aux_times,x_values,aux_p,aux_v,aux_a,aux_s)
            #normal_loss,v_loss,a_loss,stress_loss,boundary_loss,wave_loss,energy=self.get_loss(times,x_values,p,v,a,s)
            allignment_loss=self.loss_function(p,aux_p)+self.loss_function(v,aux_v)+self.loss_function(a,aux_a)+self.loss_function(s,aux_s)
            total_loss=allignment_loss*1+normal_loss*1

            total_loss.backward()
            AUX_optimizer.step()
            with torch.autograd.no_grad():
                if epoch % 100 == 0:
                    print(epoch, "AUX loss ", total_loss.data,"allignment_loss",allignment_loss.data,"normal_loss",normal_loss.data)#,"V",v_loss.data,"A",a_loss.data,"Stress",stress_loss.data,"boundary",boundary_loss.data,"wave",wave_loss,"energy",energy)
                losses[epoch] = total_loss.data
        return AUX
    def get_ouptuts(self, times, x_values, PINN):
        outputs = PINN(times, x_values)
        p = outputs[:, 0] * torch.flatten(x_values) * torch.flatten(times)+torch.flatten(x_values)
        v = outputs[:, 1] * torch.flatten(x_values) * torch.flatten(times)
        a = outputs[:, 2] * torch.flatten(x_values)
        s = outputs[:, 3]
        return p, v, a, s
    def get_aux_ouptuts(self, times, x_values, AUX):
        outputs = AUX(times, x_values)
        p = outputs[:, 0] * torch.flatten(x_values)
        v = outputs[:, 1] 
        a = outputs[:, 2] 
        s = outputs[:, 3] 
        return p, v, a, s
    def get_continuation_ouptuts(self,times,x_values,PINN,AUX):
        outputs = PINN(times, x_values)#*0 ################################################# Change needed ##############################
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
        boundary_force=(self.f+s[:self.offset])+(self.m*torch.flatten(a)[:self.offset]) #TODO look at signs here
        boundary_loss=self.loss_function(boundary_force,torch.zeros_like(boundary_force))*1 #TODO assess if this boundary loss is a good solution to our problem of different physics at the boundary
        #kinetic=2*self.m*torch.flatten(v)*torch.flatten(v)#TODO: Where did the 1/2 go?
        #spring=.5*torch.flatten(s)*torch.flatten(strain)
        #total_energy=torch.mean(kinetic+spring)
        #inital_energy=torch.tensor((.5*self.k*self.offset+1/4*self.k*(self.num_times-2*self.offset))/self.num_times).to(device)
        #inital_energy=torch.tensor(.5*self.k).to(device)
        #energy_loss=self.loss_function(total_energy,inital_energy)*0
        wave_equation=(-torch.flatten(stress_gradient) + self.m * torch.flatten(a))[self.offset:] #we slice the loss becasue the wave equation does not hold when there is an external force
        wave_loss=self.loss_function(wave_equation,torch.zeros_like(wave_equation))*1
        stress_loss = self.loss_function(stress_DNN, s) * 1
        v_loss = self.loss_function(v, torch.flatten(v_ext)) * 1
        a_loss = self.loss_function(a, torch.flatten(a_ext)) * 1


        total_loss = boundary_loss + v_loss + a_loss + stress_loss+wave_loss#+energy_loss
        return total_loss, v_loss, a_loss, stress_loss, boundary_loss, wave_loss#,energy_loss
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

    def plot(self,PINN):
        num_points=101
        x=torch.linspace(0,self.l,num_points,requires_grad=True).to(device)
        t=torch.linspace(0, 1, num_points, requires_grad=True).to(device)
        gridx,gridt=torch.meshgrid(x,t,indexing='ij')
        p,v,a,s,m=self.get_ouptuts(gridt,gridx,PINN)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(gridx.data.cpu().detach().numpy(),gridt.data.cpu().detach().numpy(),p.data.cpu().detach().numpy())
        ax.set_xlabel('X ')
        ax.set_ylabel('Time')
        ax.set_zlabel('Position')
        plt.show()
    def full_run(self,num_models,end_time):
        interval=end_time/num_models

        models=[]
        aux=[]
        start_model=PINN.run(interval)
        models.append(start_model)
        torch.save(start_model.state_dict(), "start_model.pt")
        start_aux=PINN.train_AUX(start_model,interval,old_aux=None)
        aux.append(start_aux)
        torch.save(start_aux.state_dict(), "start_aux.pt")
        for i in range(num_models-1):
            mpinn=PINN.continuation(interval,aux[-1])
            mAux=PINN.train_AUX(mpinn,interval,old_aux=aux[-1])
            models.append(mpinn)
            aux.append(mAux)
            torch.save(mpinn.state_dict(), "model"+str(i+1)+".pt")
            torch.save(mAux.state_dict(), "aux"+str(i+1)+".pt")


m=1
k=2
l=1
f=0
PINN=PINN(m,k,l,f)
run=False
aux=False
mp1=False
aux2=False
mp2=False

t0=time.time()

PINN.full_run(num_models=3,end_time=9)


if run:
    mpinn0=PINN.run(.25)
    torch.save(mpinn0.state_dict(), "mpinn0.pt")
if aux:
    mpinn0=PINN_Net().to(device)
    mpinn0.load_state_dict(torch.load("mpinn0.pt"))
    mAux0=PINN.train_AUX(mpinn0,.25,old_aux=None)
    torch.save(mAux0.state_dict(), "mAux0.pt")
if mp1:
    mAux0=PINN_Net().to(device)
    mAux0.load_state_dict(torch.load("mAux0.pt"))
    mpinn1=PINN.continuation(.25,mAux0)
    torch.save(mpinn1.state_dict(), "mpinn1.pt")
if aux2:
    mAux0=PINN_Net().to(device)
    mpinn1=PINN_Net().to(device)
    mAux0.load_state_dict(torch.load("mAux0.pt"))
    mpinn1.load_state_dict(torch.load("mpinn1.pt"))
    mAux1=PINN.train_AUX(mpinn1,.25,old_aux=mAux0)
    torch.save(mAux1.state_dict(), "mAux1.pt")
if mp2:
    mAux1=PINN_Net().to(device)
    mAux1.load_state_dict(torch.load("mAux1.pt"))
    mpinn2=PINN.continuation(.25,mAux1)
    torch.save(mpinn2.state_dict(), "mpinn2.pt")
t1=time.time()
print("Elapsed Compute Time: %5.2F"%(t1-t0))

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
        self.hidden_layer1 = nn.Linear(4, a)
        self.hidden_layer2 = nn.Linear(a, a)
        self.hidden_layer3 = nn.Linear(a, a)
        self.hidden_layer4 = nn.Linear(a, a)
        self.hidden_layer5 = nn.Linear(a, a)
        self.hidden_layer6 = nn.Linear(a, a)
        self.hidden_layer7 = nn.Linear(a, a)
        self.hidden_layer8 = nn.Linear(a, a)
        self.output_layer = nn.Linear(a, 16)
        #self.activation = torch.relu
        #self.activation=torch.nn.LeakyReLU()
        self.activation=torch.tanh
    def forward(self,t,x,y,z):
        if t.dim()==1:
            t=torch.unsqueeze(t,1)
        if x.dim()==1:
            x=torch.unsqueeze(x,1)
        if y.dim()==1:
            y=torch.unsqueeze(y,1)
        if z.dim()==1:
            z=torch.unsqueeze(z,1)
        inputs = torch.cat([t,x,y,z], axis=1)  # combined two arrays of 1 columns each to one array of 2 columns
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
        a = 250 #increasing this is a great way to inprove accuracy (at the cost of compute time)
        self.hidden_layer1 = nn.Linear(4, a)
        self.hidden_layer2 = nn.Linear(a, a)
        self.hidden_layer3 = nn.Linear(a, a)
        self.hidden_layer4 = nn.Linear(a, a)
        self.hidden_layer5 = nn.Linear(a, a)
        self.hidden_layer6 = nn.Linear(a, a)
        self.hidden_layer7 = nn.Linear(a, a)
        self.hidden_layer8 = nn.Linear(a, a)
        self.output_layer = nn.Linear(a, 16)
        #self.activation = torch.relu
        #self.activation=torch.nn.LeakyReLU()
        self.activation=torch.tanh
    def forward(self,t,x,y,z):
        if t.dim()==1:
            t=torch.unsqueeze(t,1)
        if x.dim()==1:
            x=torch.unsqueeze(x,1)
        if y.dim()==1:
            y=torch.unsqueeze(y,1)
        if z.dim()==1:
            z=torch.unsqueeze(z,1)
        inputs = torch.cat([t,x,y,z], axis=1)  # combined two arrays of 1 columns each to one array of 2 columns
        layer1_out = self.activation(self.hidden_layer1(inputs))
        layer2_out = self.activation(self.hidden_layer2(layer1_out))
        layer3_out = self.activation(self.hidden_layer3(layer2_out))
        # layer4_out = self.activation(self.hidden_layer4(layer3_out))#+layer2_out
        # layer5_out = self.activation(self.hidden_layer5(layer4_out))
        # layer6_out = self.activation(self.hidden_layer6(layer5_out))+layer3_out
        output = self.output_layer(layer3_out)
        return output

class PINN():
    def __init__(self,m,k,l,f,C1):
        self.mass=m
        self.k=k
        self.f=f
        self.l=l
        self.m=m
        self.C1=C1
        self.loss_function = torch.nn.MSELoss()
        self.offset = 1000 
        self.num_times=16*self.offset+10000 #We define 16 offsets (x,y,z,t at 0 and 1,  2^4=16)
        self.num_aux=2000
        self.mu=.166
        self.lam=1
    def run(self,end):
        PINN = PINN_Net().to(device)
        PINN_optimizer=torch.optim.Rprop(PINN.parameters(), lr=.002, step_sizes=[1e-16, 10])
        iterations = 101
        losses = np.zeros(iterations)

        for epoch in range(iterations):
            PINN_optimizer.zero_grad()
            #self.boundary_points=250
            #self.internal_points=5000
            self.boundary_points=1000
            self.internal_points=10000
            x_values,y_values,z_values,times=self.setup_points(end,self.boundary_points,self.internal_points,aux=False)
            result=self.get_ouptuts(times,x_values,y_values,z_values,PINN)
            total_loss,v_loss,a_loss,stress_loss,boundary_loss,wave_loss,volume_loss,energy_loss=self.get_loss(times,x_values,y_values,z_values,result,is_aux=False)

            total_loss.backward()
            PINN_optimizer.step()
            with torch.autograd.no_grad():
                if epoch % 100 == 0 or epoch==iterations-1:
                    print(epoch, "loss ", total_loss.data,"V",v_loss.data,"A",a_loss.data,"Stress",stress_loss.data,"boundary",boundary_loss.data,"wave",wave_loss.data,"volume",volume_loss.data,"energy (not used)",energy_loss.data)
                losses[epoch] = total_loss.data
        return PINN
    def continuation(self,end,aux):
        CPinn = PINN_Net().to(device)
        #CPinn_optimizer=torch.optim.Adam(CPinn.parameters())
        CPinn_optimizer=torch.optim.Rprop(CPinn.parameters(), lr=.002)#, step_sizes=[1e-16, 10])
        iterations = 101
        losses = np.zeros(iterations)
        end = end

        for epoch in range(iterations):
            CPinn_optimizer.zero_grad()
            self.boundary_points=1000
            self.internal_points=10000
            x_values,y_values,z_values,times=self.setup_points(end,self.boundary_points,self.internal_points,aux=False)

            C_results=self.get_continuation_ouptuts(times,x_values,y_values,z_values,CPinn,aux)
            total_loss, v_loss, a_loss, constituative_loss, boundary_loss, cauchy_loss,volume_loss, energy_loss=self.get_loss(times,x_values,y_values,z_values,C_results,is_aux=False)

            total_loss.backward()
            CPinn_optimizer.step()
            with torch.autograd.no_grad():
                if epoch % 100 == 0 or epoch==iterations-1:
                    print(epoch, "Continuation loss ", total_loss.data,"V",v_loss.data,"A",a_loss.data,"Stress",constituative_loss.data,"boundary",boundary_loss.data,"wave",cauchy_loss.data,"volume",volume_loss.data, "energy (not used)",energy_loss.data)
                losses[epoch] = total_loss.data
        plt.clf()
        plt.yscale('log')
        plt.plot(losses)
        plt.savefig('continuation_training_loss.png')
        return CPinn
    def train_AUX(self,PINN,end,old_aux):
        AUX = Aux_Net().to(device)
        AUX_optimizer=torch.optim.Rprop(AUX.parameters(), lr=.002, step_sizes=[1e-16, 10])
        #AUX_optimizer=torch.optim.Adam(AUX.parameters())#, lr=.02, step_sizes=[1e-16, 10])
        iterations = 101
        losses = np.zeros(iterations)
        end = end
        for self.epoch in range(iterations):

            #this doesn't work well for some reason-- So for now we are stuck with Adam as Rprop seems to be less stable here :(

            # if self.epoch==101:
            #     AUX_optimizer=torch.optim.Rprop(AUX.parameters(), lr=.02, step_sizes=[1e-16, 10])
            epoch=self.epoch
            AUX_optimizer.zero_grad()
            self.boundary_points=1000
            self.internal_points=10000
            x_values,y_values,z_values,times,aux_times=self.setup_points(end,self.boundary_points,self.internal_points,aux=True)




            #TODO apply slicing/corrections for boundary
            #x_values[-1]=self.l
            if old_aux != None:
                PINN_results=self.get_continuation_ouptuts(times,x_values,y_values,z_values,PINN,old_aux)
            else:
                PINN_results=self.get_ouptuts(times,x_values,y_values,z_values,PINN)
            aux_results=self.get_aux_ouptuts(aux_times,x_values,y_values,z_values,AUX)

            # slice_pinn_x=PINN_results[:,0]
            # slice_aux_x=aux_results[:,0]
            # slice_pinn_y=PINN_results[:,3]
            # slice_aux_y=aux_results[:,3]
            # slice_pinn_z=PINN_results[:,6]
            # slice_aux_z=aux_results[:,6]
            # # slice_pinn_q=PINN_results[:,8:]
            # # slice_aux_q=aux_results[:,8:]
            # al1=self.loss_function(slice_pinn_x,slice_aux_x)
            # al2=self.loss_function(slice_pinn_y,slice_aux_y)
            # al3=self.loss_function(slice_pinn_z,slice_aux_z)
            # #al4=self.loss_function(slice_pinn_q,slice_aux_q)
            # allignment_loss=al1+al2+al3#+al4

            # print(slice_pinn_x.size())
            # quit()


            normal_loss,v_loss,a_loss,stress_loss,boundary_loss,wave_loss,energy,volume_loss=self.get_loss(aux_times,x_values,y_values,z_values,aux_results,is_aux=True)
            #normal_loss,v_loss,a_loss,stress_loss,boundary_loss,wave_loss,energy=self.get_loss(times,x_values,p,v,a,s)
            allignment_loss=self.loss_function(PINN_results,aux_results)
            total_loss=allignment_loss*1+normal_loss*.1 #TODO re-implement energy correction

            total_loss.backward()
            AUX_optimizer.step()
            with torch.autograd.no_grad():
                if epoch % 100 == 0 or epoch==iterations-1: 
                    print(epoch, "AUX loss ", total_loss.data,"allignment_loss",allignment_loss.data,"normal_loss",normal_loss.data)
                    print("         V",v_loss.data,"A",a_loss.data,"Stress",stress_loss.data,"boundary",boundary_loss.data,"wave",wave_loss.data,"volume",volume_loss.data,"energy",energy.data)
                losses[epoch] = total_loss.data
        plt.clf()
        plt.yscale('log')
        plt.plot(losses)
        plt.savefig('aux_training_loss.png')
        return AUX
    def get_ouptuts(self, times, x_values,y_values,z_values, PINN):
        outputs = PINN(times, x_values,y_values,z_values)
        #x position velocity and acceleration (fixed at x=0 and no motion initialy)
        px = outputs[:, 0] * torch.flatten(x_values) * torch.flatten(times)+torch.flatten(x_values)*3 #Apply inital displacement here
        vx = outputs[:, 1] * torch.flatten(x_values) * torch.flatten(times)
        ax = outputs[:, 2] * torch.flatten(x_values) 
        #y position velocity and acceleration (fixed at y=0 and no motion initialy)
        py = outputs[:, 3] * torch.flatten(y_values) * torch.flatten(times)-torch.flatten(y_values)*.5 #Apply inital displacement here
        vy = outputs[:, 4] * torch.flatten(y_values) * torch.flatten(times)
        ay = outputs[:, 5] * torch.flatten(y_values) 
        #z position velocity and acceleration (fixed at z=0 and no motion initialy)
        pz = outputs[:, 6] * torch.flatten(z_values) * torch.flatten(times)-torch.flatten(z_values)*.5 #Apply inital displacement here
        vz = outputs[:, 7] * torch.flatten(z_values) * torch.flatten(times)
        az = outputs[:, 8] * torch.flatten(z_values) 
        #6D stress tensor (are there well defined BC's here??)
        sxx = outputs[:, 9]
        sxy = outputs[:, 10]
        sxz = outputs[:, 11]
        syy = outputs[:, 12]
        syz = outputs[:, 13]
        szz = outputs[:, 14]
        #incompressible to enforce incompressability
        incompressible=outputs[:,15]
        #put all the outputs back into 1 tensor so they are easier to move around
        out=torch.stack((px,vx,ax,py,vy,ay,pz,vz,az,sxx,sxy,sxz,syy,syz,szz,incompressible),dim=1)
        return out
    def get_aux_ouptuts(self, times, x_values,y_values,z_values, AUX):
        outputs = AUX(times, x_values,y_values,z_values)
        #x position velocity and acceleration (fixed at x=0)
        px = outputs[:, 0] * torch.flatten(x_values)  
        vx = outputs[:, 1] * torch.flatten(x_values) 
        ax = outputs[:, 2] * torch.flatten(x_values) 
        #y position velocity and acceleration (fixed at y=0)
        py = outputs[:, 3] * torch.flatten(y_values) 
        vy = outputs[:, 4] * torch.flatten(y_values)
        ay = outputs[:, 5] * torch.flatten(y_values)
        #z position velocity and acceleration (fixed at z=0)
        pz = outputs[:, 6] * torch.flatten(z_values) 
        vz = outputs[:, 7] * torch.flatten(z_values)
        az = outputs[:, 8] * torch.flatten(z_values)
        #6D stress tensor (are there well defined BC's here??)
        sxx = outputs[:, 9]
        sxy = outputs[:, 10]
        sxz = outputs[:, 11]
        syy = outputs[:, 12]
        syz = outputs[:, 13]
        szz = outputs[:, 14]
        #incompressible to enforce incompressability
        incompressible=outputs[:,15]
        #put all the outputs back into 1 tensor so they are easier to move around
        out=torch.stack((px,vx,ax,py,vy,ay,pz,vz,az,sxx,sxy,sxz,syy,syz,szz,incompressible),dim=1)
        return out
    def get_continuation_ouptuts(self,times,x_values,y_values,z_values,PINN,AUX):
        outputs = PINN(times, x_values,y_values,z_values)
        aux_times=torch.zeros_like(times)
        aux_outputs=self.get_aux_ouptuts(aux_times,x_values,y_values,z_values,AUX)
        #remove gradient information from AUX
        aux_outputs.grad=torch.zeros_like(aux_outputs)
        #x position velocity and acceleration (fixed at x=0 and no motion initialy)
        px = aux_outputs[:,0]+outputs[:, 0] * torch.flatten(x_values) * torch.flatten(times)
        vx = aux_outputs[:,1]+outputs[:, 1] * torch.flatten(x_values) * torch.flatten(times)
        ax = aux_outputs[:,2]+outputs[:, 2] * torch.flatten(x_values) * torch.flatten(times)
        #y position velocity and acceleration (fixed at y=0 and no motion initialy)
        py = aux_outputs[:,3]+outputs[:, 3] * torch.flatten(y_values) * torch.flatten(times)
        vy = aux_outputs[:,4]+outputs[:, 4] * torch.flatten(y_values) * torch.flatten(times)
        ay = aux_outputs[:,5]+outputs[:, 5] * torch.flatten(y_values) * torch.flatten(times)
        #z position velocity and acceleration (fixed at z=0 and no motion initialy)
        pz = aux_outputs[:,6]+outputs[:, 6] * torch.flatten(z_values) * torch.flatten(times)
        vz = aux_outputs[:,7]+outputs[:, 7] * torch.flatten(z_values) * torch.flatten(times)
        az = aux_outputs[:,8]+outputs[:, 8] * torch.flatten(z_values) * torch.flatten(times)
        #6D stress tensor (are there well defined BC's here??)
        sxx = aux_outputs[:,9]+outputs[:, 9]                          * torch.flatten(times)
        sxy = aux_outputs[:,10]+outputs[:, 10]                        * torch.flatten(times)
        sxz = aux_outputs[:,11]+outputs[:, 11]                        * torch.flatten(times)
        syy = aux_outputs[:,12]+outputs[:, 12]                        * torch.flatten(times)
        syz = aux_outputs[:,13]+outputs[:, 13]                        * torch.flatten(times)
        szz = aux_outputs[:,14]+outputs[:, 14]                        * torch.flatten(times)
        #incompressible to enforce incompressability
        incompressible=aux_outputs[:,15]+outputs[:,15]                * torch.flatten(times)
        #put all the outputs back into 1 tensor so they are easier to move around
        out=torch.stack((px,vx,ax,py,vy,ay,pz,vz,az,sxx,sxy,sxz,syy,syz,szz,incompressible),dim=1)
        return out
    def get_loss(self, times, x_values,y_values,z_values, result,is_aux):
        #note that in this formulation we still assume sxz=szx (no rotational acceleration I think)
        #extract outputs
        result_inner=result #TODO add slicing operation so that we can split the insides from the boundaries (where the stress divergence is and is not applicable)
        px = result_inner[:, 0]
        vx = result_inner[:, 1]
        ax = result_inner[:, 2]
        py = result_inner[:, 3] 
        vy = result_inner[:, 4]
        ay = result_inner[:, 5]
        pz = result_inner[:, 6]
        vz = result_inner[:, 7]
        az = result_inner[:, 8]
        sxx = result_inner[:, 9]
        sxy = result_inner[:, 10]
        sxz = result_inner[:, 11]
        syy = result_inner[:, 12]
        syz = result_inner[:, 13]
        szz = result_inner[:, 14]
        incompressible=result_inner[:,15]
        t_mask=torch.flatten(torch.cat((torch.ceil(times),torch.ceil(times),torch.ceil(times)),dim=0))


        points=px.size()[0]
        #maybe this tensor needs to be (points, 1) in shape...
        incompressible=torch.reshape(incompressible,(points,1,1))
        ones=torch.ones((points),requires_grad=True).to(device)

        v_x = torch.autograd.grad(px, times, grad_outputs=ones, create_graph=True)[0]
        a_x = torch.autograd.grad(vx, times, grad_outputs=ones, create_graph=True)[0]
        v_y = torch.autograd.grad(py, times, grad_outputs=ones, create_graph=True)[0]
        a_y = torch.autograd.grad(vy, times, grad_outputs=ones, create_graph=True)[0]
        v_z = torch.autograd.grad(pz, times, grad_outputs=ones, create_graph=True)[0]
        a_z = torch.autograd.grad(vz, times, grad_outputs=ones, create_graph=True)[0]

        #This section adapted from "Pinn_optimizaiton/triaxial/Traixial_no_BCnet.py" on windows laptop
        #stress divergence
        stress_divergence_x = torch.autograd.grad(sxx, x_values,grad_outputs=ones, create_graph=True)[0]
        stress_divergence_y = torch.autograd.grad(syy, y_values,grad_outputs=ones, create_graph=True)[0]
        stress_divergence_z = torch.autograd.grad(szz, z_values,grad_outputs=ones, create_graph=True)[0]
        #shear divergence
        shear_divergence_xy = torch.autograd.grad(sxy, y_values,grad_outputs=ones, create_graph=True)[0]
        shear_divergence_xz = torch.autograd.grad(sxz, z_values,grad_outputs=ones, create_graph=True)[0]
        shear_divergence_yz = torch.autograd.grad(syz, z_values,grad_outputs=ones, create_graph=True)[0]
        shear_divergence_yx = torch.autograd.grad(sxy, x_values,grad_outputs=ones, create_graph=True)[0]
        shear_divergence_zx = torch.autograd.grad(sxz, x_values,grad_outputs=ones, create_graph=True)[0]
        shear_divergence_zy = torch.autograd.grad(syz, y_values,grad_outputs=ones, create_graph=True)[0]
        #strain variables
        strain_x = torch.autograd.grad(px, x_values,grad_outputs=ones, create_graph=True)[0]
        strain_y = torch.autograd.grad(py, y_values,grad_outputs=ones, create_graph=True)[0]
        strain_z = torch.autograd.grad(pz, z_values,grad_outputs=ones, create_graph=True)[0]
        strain_xy = torch.autograd.grad(px, y_values,grad_outputs=ones, create_graph=True)[0]
        strain_xz = torch.autograd.grad(px, z_values,grad_outputs=ones, create_graph=True)[0]
        strain_yz = torch.autograd.grad(py, z_values,grad_outputs=ones, create_graph=True)[0]
        strain_yx = torch.autograd.grad(py, x_values,grad_outputs=ones, create_graph=True)[0]
        strain_zx = torch.autograd.grad(pz, x_values,grad_outputs=ones, create_graph=True)[0]
        strain_zy = torch.autograd.grad(pz, y_values,grad_outputs=ones, create_graph=True)[0]

        #We do some indexing in these next sections to partition all of the points we get into the sections that are on the domain boundary, and those that are internal. 
        #My addition/modification to the cauchy momentum equation/wave equaiton (only valid on the interior of the domain)
        div_x=(-1*(stress_divergence_x+shear_divergence_xy+shear_divergence_xz)+self.m*torch.flatten(ax))[4*self.boundary_points:-4*self.boundary_points]
        div_y=(-1*(stress_divergence_y+shear_divergence_yz+shear_divergence_yx)+self.m*torch.flatten(ay))[4*self.boundary_points:-4*self.boundary_points]
        div_z=(-1*(stress_divergence_z+shear_divergence_zx+shear_divergence_zy)+self.m*torch.flatten(az))[4*self.boundary_points:-4*self.boundary_points]
        #Boundary loss (only valid on the boundary of the domain (where the value is 1))
        boundary_x=(1*(sxx+sxy+sxz)+self.m*torch.flatten(ax))[4*self.boundary_points+self.internal_points:-3*self.boundary_points]
        boundary_y=(1*(sxy+syy+syz)+self.m*torch.flatten(ay))[5*self.boundary_points+self.internal_points:-2*self.boundary_points]
        boundary_z=(1*(sxz+syz+szz)+self.m*torch.flatten(az))[6*self.boundary_points+self.internal_points:-1*self.boundary_points]



        # print(z_values[6*self.boundary_points+self.internal_points:-1*self.boundary_points])
        # quit()
        #Compute linear stress from dispacements
        eye1 = [[[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]]
        eye1 = torch.tensor(eye1, device=device)
        eye = eye1.repeat(points, 1, 1)
        incompress = eye * incompressible
        x_components = torch.stack((strain_x, strain_xy, strain_xz), dim=1)
        y_components = torch.stack((strain_yx, strain_y, strain_yz), dim=1)
        z_components = torch.stack((strain_zx, strain_zy, strain_z), dim=1)
        F = torch.stack((x_components, y_components, z_components), dim=2)  #Is this correct????  further down we do: strain_tensor=torch.flatten(torch.stack((x_strain,y_strain,z_strain),dim=2),start_dim=2)


        F=F+eye
        Ft = torch.transpose(F, 1, 2)
        B = torch.matmul(F, Ft)
        C = torch.matmul(Ft,F)
        I1=C.diagonal(offset=0,dim1=-1,dim2=-2).sum(-1)
        stress=2*self.C1*B-incompress

        # x_strain=torch.stack((strain_x,strain_xy,strain_xz),dim=1)
        # y_strain=torch.stack((strain_yx,strain_y,strain_yz),dim=1)
        # z_strain=torch.stack((strain_zx,strain_zy,strain_z),dim=1)
        # strain_t=strain_x+strain_y+strain_z
        # strain_trace=torch.reshape(strain_t,(points,1,1)) #We can proabably do this operation more efficie
        # strain_tensor=torch.flatten(torch.stack((x_strain,y_strain,z_strain),dim=2),start_dim=2)
        #TODO Apply stresses properly ######################################################
        #stress = 2*self.mu*strain_tensor+self.lam*strain_trace*eye 
        #stress=self.k*strain_tensor

        #Obtain stress directly from DNN output
        stress_x = torch.reshape(sxx, (points, 1))
        stress_y = torch.reshape(syy, (points, 1))
        stress_z = torch.reshape(szz, (points, 1))
        shear_xy = torch.reshape(sxy, (points, 1))
        shear_xz = torch.reshape(sxz, (points, 1))
        shear_yz = torch.reshape(syz, (points, 1))
        x_comp = torch.cat((stress_x, shear_xy, shear_xz), dim=1)
        y_comp = torch.cat((shear_xy, stress_y, shear_yz), dim=1)
        z_comp = torch.cat((shear_xz, shear_yz, stress_z), dim=1)
        stress_DNN = torch.stack((x_comp, y_comp, z_comp), dim=2)

        #Begin Computing losses
        volume=torch.linalg.det(F)
        volume_ones = torch.ones_like(volume)
        boundary_motion=torch.cat((boundary_x,boundary_y,boundary_z))
        boundary_zeros=torch.zeros_like(boundary_motion)
        cauchy_momentum = torch.cat((div_x, div_y, div_z))
        cauchy_zeros = torch.zeros_like(cauchy_momentum)

        computed_v=torch.cat((v_x,v_y,v_z))
        given_v=torch.cat((vx,vy,vz))
        computed_a=torch.cat((a_x,a_y,a_z))
        given_a=torch.cat((ax,ay,az))

        mse_cost_function = torch.nn.MSELoss()

        cauchy_loss = mse_cost_function(cauchy_momentum, cauchy_zeros)
        constituative_loss = mse_cost_function(stress, stress_DNN)
        volume_loss=mse_cost_function(volume,volume_ones)*10 
        #maybe we can remove the velocity and acceleration losses 
        velocity_loss=mse_cost_function(torch.flatten(computed_v),given_v)
        acceleration_loss=mse_cost_function(torch.flatten(computed_a),given_a)
        # velocity_loss=mse_cost_function(torch.flatten(computed_v)*t_mask,given_v*t_mask)
        # acceleration_loss=mse_cost_function(torch.flatten(computed_a)*t_mask,given_a*t_mask)
        boundary_loss=mse_cost_function(boundary_motion,boundary_zeros)

        if is_aux:
            velocity_loss=torch.tensor(0)
            acceleration_loss=torch.tensor(0)
            #compute energy loss
            #kinetic energy=1/2*m*v^2 
            #in 1D we used 2 rather than 1/2
            kinetic_x=.5*self.m*torch.square(torch.flatten(vx))
            kinetic_y=.5*self.m*torch.square(torch.flatten(vy))
            kinetic_z=.5*self.m*torch.square(torch.flatten(vz))
            total_kinetic=kinetic_x+kinetic_y+kinetic_z
            #spring potential for linearly elastic, isotropic materials under small strains
            spring=self.C1*(I1-3)
            #spring=.5*(sxx*torch.flatten(strain_x)+syy*torch.flatten(strain_y)+szz*torch.flatten(strain_z))+1*(sxy*torch.flatten(strain_xy)+sxz*torch.flatten(strain_xz)+syz*torch.flatten(strain_yz))
            total_energy=total_kinetic+spring
            #inital_energy=torch.tensor(.5*self.k).to(device)# this should come from strain energy density function
            inital_energy=torch.tensor(self.C1*(16.5-3)).to(device) #assume inital strech ratios are 4,.5,.5, so the sum of squares, (I1) = 16+.25+.25=16.5
            #energy_loss=self.loss_function(total_energy,inital_energy*torch.ones_like(total_energy))*1 #this enforces local energy conservation
            energy_loss=self.loss_function(torch.mean(total_energy),torch.mean(inital_energy*torch.ones_like(total_energy)))*1 #this enforces global energy conservation
            #compute entire loss
            total_loss = boundary_loss + velocity_loss + acceleration_loss + constituative_loss+cauchy_loss+energy_loss+volume_loss
            return total_loss, velocity_loss, acceleration_loss, constituative_loss, boundary_loss, cauchy_loss,energy_loss,volume_loss
        else:
            kinetic_x=.5*self.m*torch.square(torch.flatten(vx))
            kinetic_y=.5*self.m*torch.square(torch.flatten(vy))
            kinetic_z=.5*self.m*torch.square(torch.flatten(vz))
            total_kinetic=kinetic_x+kinetic_y+kinetic_z
            #spring potential for linearly elastic, isotropic materials under small strains
            spring=self.C1*(I1-3)
            #spring=.5*(sxx*torch.flatten(strain_x)+syy*torch.flatten(strain_y)+szz*torch.flatten(strain_z))+1*(sxy*torch.flatten(strain_xy)+sxz*torch.flatten(strain_xz)+syz*torch.flatten(strain_yz))
            total_energy=total_kinetic+spring
            #inital_energy=torch.tensor(.5*self.k).to(device)# this should come from strain energy density function
            inital_energy=torch.tensor(self.C1*(16.5-3)).to(device) #assume inital strech ratios are 4,.5,.5, so the sum of squares, (I1) = 16+.25+.25=16.5


            #energy_loss=self.loss_function(total_energy,inital_energy*torch.ones_like(total_energy))*1 #this enforces local energy conservation
            energy_loss=self.loss_function(torch.mean(total_energy),torch.mean(inital_energy*torch.ones_like(total_energy)))*1 #this enforces global energy conservation
            #We compute energy loss, but not 


            total_loss = boundary_loss + velocity_loss + acceleration_loss + constituative_loss+cauchy_loss+volume_loss+energy_loss*1
            return total_loss, velocity_loss, acceleration_loss, constituative_loss, boundary_loss, cauchy_loss,volume_loss,energy_loss
    def setup_points(self,end,boundary_points,internal_points,aux):
        zeros=torch.zeros(boundary_points,requires_grad=True)
        ones=torch.ones(boundary_points,requires_grad=True)

        num_points = 10
        num_times = 10
        x = torch.linspace(0, self.l, num_points, requires_grad=True).to(device)
        y = torch.linspace(0, self.l, num_points, requires_grad=True).to(device)
        z = torch.linspace(0, self.l, num_points, requires_grad=True).to(device)
        t = torch.linspace(0, end, num_times,requires_grad=True).to(device)
        # t=torch.linspace(0, duration, num_times, requires_grad=True).to(device)

        gridx, gridy, gridz, gridt = torch.meshgrid(x, y, z, t, indexing='ij')

        gridx = torch.flatten(gridx)
        gridy = torch.flatten(gridy)
        gridz = torch.flatten(gridz)
        gridt = torch.flatten(gridt)

        x_points=torch.cat((zeros,torch.rand(3*boundary_points+internal_points),ones,torch.rand(3*boundary_points))).to(device)
        y_points=torch.cat((torch.rand(boundary_points),zeros,torch.rand(3*boundary_points+internal_points),ones,torch.rand(2*boundary_points))).to(device)
        z_points=torch.cat((torch.rand(2*boundary_points),zeros,torch.rand(3*boundary_points+internal_points),ones,torch.rand(1*boundary_points))).to(device)
        t_points=torch.cat((torch.rand(3*boundary_points),zeros,torch.rand(3*boundary_points+internal_points),ones)).to(device)*end
        if aux:
            t_points=torch.ones_like(x_points)*end
            aux_t_points=torch.zeros_like(x_points,requires_grad=True)
            return x_points,y_points,z_points,t_points,aux_t_points
        return x_points,y_points,z_points,t_points



    def full_run(self,num_models,end_time):
        interval=end_time/num_models
        models=[]
        aux=[]
        mpinn=PINN.run(interval)
        models.append(mpinn)
        torch.save(mpinn.state_dict(), "3D_start_model.pt")
        mAux=PINN.train_AUX(mpinn,interval,old_aux=None)

        # start_aux=PINN_Net().to(device)
        # start_aux.load_state_dict(torch.load("3D_start_aux.pt"))

        aux.append(mAux)
        torch.save(mAux.state_dict(), "3D_start_aux.pt")
        for i in range(num_models-1):
            mpinn=PINN.continuation(interval,mAux)
            if i==num_models-2:
                torch.save(mpinn.state_dict(), "3D_model"+str(i+1)+".pt")
                break
            mAux=PINN.train_AUX(mpinn,interval,old_aux=mAux)
            models.append(mpinn)
            aux.append(mAux)
            torch.save(mpinn.state_dict(), "3D_model"+str(i+1)+".pt")
            torch.save(mAux.state_dict(), "3D_aux"+str(i+1)+".pt")


m=1
k=2
l=1
f=0
C1=.166
PINN=PINN(m,k,l,f,C1)
run=False
aux=False
mp1=False
aux2=False
mp2=False

t0=time.time()

PINN.full_run(num_models=16,end_time=8)
#for 3D we seem to be stable at ~.5 seconds per model
#1 second/model seems insufficient for 3 layers of 250

if run:
    mpinn0=PINN.run(1.5)
    torch.save(mpinn0.state_dict(), "mpinn0.pt")
if aux:
    mpinn0=PINN_Net().to(device)
    mpinn0.load_state_dict(torch.load("mpinn0.pt"))
    mAux0=PINN.train_AUX(mpinn0,1.5,old_aux=None)
    torch.save(mAux0.state_dict(), "mAux0.pt")
if mp1:
    mAux0=Aux_Net().to(device)
    mAux0.load_state_dict(torch.load("mAux0.pt"))
    mpinn1=PINN.continuation(1.5,mAux0)
    torch.save(mpinn1.state_dict(), "mpinn1.pt")
if aux2:
    mAux0=Aux_Net().to(device)
    mpinn1=PINN_Net().to(device)
    mAux0.load_state_dict(torch.load("mAux0.pt"))
    mpinn1.load_state_dict(torch.load("mpinn1.pt"))
    mAux1=PINN.train_AUX(mpinn1,1.5,old_aux=mAux0)
    torch.save(mAux1.state_dict(), "mAux1.pt")
if mp2:
    mAux1=Aux_Net().to(device)
    mAux1.load_state_dict(torch.load("mAux1.pt"))
    mpinn2=PINN.continuation(.25,mAux1)
    torch.save(mpinn2.state_dict(), "mpinn2.pt")
t1=time.time()
print("Elapsed Compute Time: %5.2F"%(t1-t0))

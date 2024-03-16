import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

class PINN_Net(nn.Module):
    def __init__(self):
        super(PINN_Net, self).__init__()
        a = 100 #increasing this is a great way to inprove accuracy (at the cost of compute time)
        self.hidden_layer1 = nn.Linear(3, a)
        self.hidden_layer2 = nn.Linear(a, a)
        self.hidden_layer3 = nn.Linear(a, a)
        self.hidden_layer4 = nn.Linear(a, a)
        self.hidden_layer5 = nn.Linear(a, a)
        self.hidden_layer6 = nn.Linear(a, a)
        # self.hidden_layer7 = nn.Linear(a, a)
        # self.hidden_layer8 = nn.Linear(a, a)
        self.output_layer = nn.Linear(a, 10)
        #self.activation = torch.relu
        #self.activation=torch.nn.LeakyReLU()
        self.activation=torch.tanh
    def forward(self,t,x,y):
        if t.dim()==1:
            t=torch.unsqueeze(t,1)
        if x.dim()==1:
            x=torch.unsqueeze(x,1)
        if y.dim()==1:
            y=torch.unsqueeze(y,1)
        inputs = torch.cat([t,x,y], axis=1)  # combined two arrays of 1 columns each to one array of 2 columns
        layer1_out = self.activation(self.hidden_layer1(inputs))
        layer2_out = self.activation(self.hidden_layer2(layer1_out))
        layer3_out = self.activation(self.hidden_layer3(layer2_out))
        layer4_out = self.activation(self.hidden_layer4(layer3_out))+layer2_out
        layer5_out = self.activation(self.hidden_layer5(layer4_out))
        layer6_out = self.activation(self.hidden_layer6(layer5_out))+layer3_out
        # layer7_out = self.activation(self.hidden_layer7(layer6_out))
        # layer8_out = self.activation(self.hidden_layer8(layer7_out))
        output = self.output_layer(layer6_out)
        return output

class Aux_Net(nn.Module):
    def __init__(self):
        super(Aux_Net, self).__init__()
        a = 100 #increasing this is a great way to inprove accuracy (at the cost of compute time)
        self.hidden_layer1 = nn.Linear(3, a)
        self.hidden_layer2 = nn.Linear(a, a)
        self.hidden_layer3 = nn.Linear(a, a)
        self.hidden_layer4 = nn.Linear(a, a)
        self.hidden_layer5 = nn.Linear(a, a)
        self.hidden_layer6 = nn.Linear(a, a)
        # self.hidden_layer7 = nn.Linear(a, a)
        # self.hidden_layer8 = nn.Linear(a, a)
        self.output_layer = nn.Linear(a, 10)
        #self.activation = torch.relu
        #self.activation=torch.nn.LeakyReLU()
        self.activation=torch.tanh
    def forward(self,t,x,y):
        if t.dim()==1:
            t=torch.unsqueeze(t,1)
        if x.dim()==1:
            x=torch.unsqueeze(x,1)
        if y.dim()==1:
            y=torch.unsqueeze(y,1)
        inputs = torch.cat([t,x,y], axis=1)  # combined two arrays of 1 columns each to one array of 2 columns
        layer1_out = self.activation(self.hidden_layer1(inputs))
        layer2_out = self.activation(self.hidden_layer2(layer1_out))
        layer3_out = self.activation(self.hidden_layer3(layer2_out))
        layer4_out = self.activation(self.hidden_layer4(layer3_out))+layer2_out
        layer5_out = self.activation(self.hidden_layer5(layer4_out))
        layer6_out = self.activation(self.hidden_layer6(layer5_out))+layer3_out
        output = self.output_layer(layer6_out)
        return output

class PINN():
    def __init__(self,m,k,l,f):
        self.mass=m
        self.k=k
        self.f=f
        self.l=l
        self.m=m
        self.loss_function = torch.nn.MSELoss()
        self.offset0d=1
        self.offset1d=20
        self.offset2d=self.offset1d*self.offset1d
        self.offset3d=self.offset1d*self.offset1d*self.offset1d
        self.start_offset=4*self.offset0d+4*self.offset1d+self.offset2d
        self.normal_iteraitons=60
        self.aux_iterations=40

        eye1 = [[[1, 0],
               [0, 1]]]
        eye1 = torch.tensor(eye1, device=device)
        self.eye = eye1.repeat(self.offset0d*8+self.offset1d*12+self.offset2d*6+self.offset3d, 1, 1)
    def run(self,end):
        PINN = PINN_Net().to(device)
        PINN_optimizer=torch.optim.Rprop(PINN.parameters(), lr=.001, step_sizes=[1e-16, 10])
        iterations = self.normal_iteraitons
        losses = np.zeros(iterations)
        end = end

        # times,x_values,y_values=self.get_points(end,aux=False)
        # self.x_mask=torch.flatten(torch.floor(x_values))
        # self.y_mask=torch.flatten(torch.floor(y_values))

        for epoch in range(iterations):
            PINN_optimizer.zero_grad()

            times,x_values,y_values=self.get_points(end,aux=False)
            self.x_mask=torch.flatten(torch.floor(x_values))
            self.y_mask=torch.flatten(torch.floor(y_values))

            outputs=self.get_ouptuts(times,x_values,y_values,PINN)
            total_loss,v_loss,a_loss,stress_loss,boundary_loss,wave_loss,volume_loss=self.get_loss(times,x_values,y_values,outputs)

            total_loss.backward()
            PINN_optimizer.step()
            with torch.autograd.no_grad():
                if epoch % 100 == 0 or epoch==iterations-1:
                    # px= outputs[:, 0]
                    vx= outputs[-self.start_offset:, 1]
                    vy= outputs[-self.start_offset:, 4]
                    kin=torch.square(vx)+torch.square(vy)
                    # sxx = outputs[:, 6]
                    # sxy = outputs[:, 7]
                    # syy = outputs[:, 8]
                    # incompress=outputs[:,9]
                    # print( )
                    # # strain_x = torch.autograd.grad(px, x_values, torch.ones_like(px))[0]
                    # # strain_y=torch.autograd.grad(py, y_values, torch.ones_like(py))[0]
                    # print(sxx.data)
                    # print(syy.data)
                    # print(sxy.data)
                    # print(incompress.data)
                    # # print(strain_x.data)
                    # # print(strain_y.data)
                    print(epoch, "loss ", total_loss.data,"V",v_loss.data,"A",a_loss.data,"Stress",stress_loss.data,"boundary",boundary_loss.data,"wave",wave_loss.data,"Volume",volume_loss.data)#,"energy",energy)
                    print("kinetic energy:", torch.mean(kin*2*self.m))
                losses[epoch] = total_loss.data
        plt.clf()
        plt.yscale('log')
        plt.plot(losses)
        plt.savefig('start_loss.png')

        return PINN
    def continuation(self,end,aux):
        CPinn = PINN_Net().to(device)
        CPinn_optimizer=torch.optim.Rprop(CPinn.parameters(), lr=.001, step_sizes=[1e-16, 10])
        iterations = self.normal_iteraitons
        losses = np.zeros(iterations)
        end = end

        for epoch in range(iterations):
            CPinn_optimizer.zero_grad()
            
            times,x_values,y_values=self.get_points(end,aux=False)
            self.x_mask=torch.flatten(torch.floor(x_values))
            self.y_mask=torch.flatten(torch.floor(y_values))

            out=self.get_continuation_ouptuts(times,x_values,y_values,CPinn,aux)
            total_loss,v_loss,a_loss,stress_loss,boundary_loss,wave_loss,volume_loss=self.get_loss(times,x_values,y_values,out)

            total_loss.backward()
            CPinn_optimizer.step()
            with torch.autograd.no_grad():
                if epoch % 100 == 0 or epoch==iterations-1:
                    vx= out[-self.start_offset:, 1]
                    vy= out[-self.start_offset:, 4]
                    t1=torch.mean(times[-self.start_offset:])
                    end_kin=torch.square(vx)+torch.square(vy)
                    vx= out[:self.start_offset, 1]
                    vy= out[:self.start_offset, 4]
                    t2=torch.mean(times[:self.start_offset])
                    start_kin=torch.square(vx)+torch.square(vy)

                    print("end_kin", torch.mean(end_kin*2*self.m),"start_kin", torch.mean(start_kin*2*self.m))
                    print(epoch, "Continuation loss ", total_loss.data)#,"V",v_loss.data,"A",a_loss.data,"Stress",stress_loss.data,"boundary",boundary_loss.data,"wave",wave_loss,"energy",energy)
                losses[epoch] = total_loss.data
        plt.clf()
        plt.yscale('log')
        plt.plot(losses)
        plt.savefig('continuation_loss.png')
        return CPinn
    def train_AUX(self,PINN,end,old_aux):
        time_offset=0.0
        start=end-time_offset
        AUX = Aux_Net().to(device)
        AUX_optimizer=torch.optim.Rprop(AUX.parameters(), lr=.001, step_sizes=[1e-16, 100])
        iterations = self.aux_iterations
        losses = np.zeros(iterations)
        end = end
        for epoch in range(iterations):
            AUX_optimizer.zero_grad()
            
            times, aux_times, x_values,y_values=self.get_points(end,aux=True)
            self.x_mask=torch.flatten(torch.floor(x_values))
            self.y_mask=torch.flatten(torch.floor(y_values))

            if old_aux != None:
                prior_outputs=self.get_continuation_ouptuts(times,x_values,y_values,PINN,old_aux)
                #print("continuation")
            else:
                prior_outputs=self.get_ouptuts(times,x_values,y_values,PINN)
                #print("start")
            aux_outputs=self.get_aux_ouptuts(aux_times,x_values,y_values,AUX)

            px=prior_outputs[:,0]
            vx=prior_outputs[:,1]
            #ax=prior_outputs[:,2]
            #sxx=prior_outputs[:,6]
            py = prior_outputs[:, 3]
            vy = prior_outputs[:, 4]

            aux_px=aux_outputs[:,0]
            aux_vx=aux_outputs[:,1]
            #aux_ax=aux_outputs[:,2]
            #aux_sxx=aux_outputs[:,6]
            aux_py = aux_outputs[:, 3]
            aux_vy = aux_outputs[:, 4]


            normal_loss,stress_loss,boundary_loss,wave_loss,volume,energy=self.get_AUX_loss(aux_times,x_values,y_values,aux_outputs)
            #normal_loss,v_loss,a_loss,stress_loss,boundary_loss,wave_loss,energy=self.get_loss(times,x_values,p,v,a,s)
            #allignment_loss=self.loss_function(px,aux_px)+self.loss_function(vx,aux_vx)+self.loss_function(vy,aux_vy)+self.loss_function(py,aux_py)
            allignment_loss=self.loss_function(prior_outputs[:,:-1],aux_outputs[:,:-1])#we do not need to match incompressible
            if epoch<=5:
                total_loss=allignment_loss*1
            else:
                total_loss=allignment_loss*1+normal_loss*.1
            total_loss.backward()
            AUX_optimizer.step()
            with torch.autograd.no_grad():
                if epoch % 100 == 0 or epoch==iterations-1:

                    vx= prior_outputs[-self.start_offset:, 1]
                    vy= prior_outputs[-self.start_offset:, 4]
                    t1=torch.mean(times[-self.start_offset:])
                    prior_kin=torch.square(vx)+torch.square(vy)

                    v_auxx= aux_outputs[-self.start_offset:, 1]
                    v_auxy= aux_outputs[-self.start_offset:, 4]
                    t2=torch.mean(aux_times[-self.start_offset:])
                    aux_kin=torch.square(v_auxx)+torch.square(v_auxy)
                    print(epoch, "AUX loss ", total_loss.data,"allignment_loss",allignment_loss.data,"normal_loss",normal_loss.data)#,self.loss_function(px,aux_px),self.loss_function(vx,aux_vx),self.loss_function(ax,aux_ax),self.loss_function(sxx,aux_sxx))#,"V",v_loss.data,"A",a_loss.data,"Stress",stress_loss.data,"boundary",boundary_loss.data,"wave",wave_loss,"energy",energy)
                    print("         stress",stress_loss.data,"boundary",boundary_loss.data,"wave",wave_loss.data,"Volume",volume.data,"energy",energy.data)
                    print("aux kin", torch.mean(aux_kin*2*self.m),"prior_kin",torch.mean(prior_kin*2*self.m))
                losses[epoch] = total_loss.data
        plt.clf()
        plt.yscale('log')
        plt.plot(losses)
        plt.savefig('aux_loss.png')

        return AUX
    def get_ouptuts(self, times, x_values,y_values, PINN):
        outputs = PINN(times, x_values,y_values)
        px = outputs[:, 0] * torch.flatten(x_values) * torch.flatten(times)+torch.flatten(x_values)*.5
        vx = outputs[:, 1] * torch.flatten(x_values) * torch.flatten(times)
        ax = outputs[:, 2] * torch.flatten(x_values)

        py = outputs[:, 3] * torch.flatten(y_values) * torch.flatten(times)-torch.flatten(y_values)*1/3
        vy = outputs[:, 4] * torch.flatten(y_values) * torch.flatten(times)
        ay = outputs[:, 5] * torch.flatten(y_values)

        sxx = outputs[:, 6]
        sxy = outputs[:, 7]
        syy = outputs[:, 8]
        incompress=outputs[:,9]
        out=torch.stack( (px,vx,ax,py,vy,ay,sxx,sxy,syy,incompress),dim=1)
        return out
    def get_aux_ouptuts(self, times, x_values,y_values, AUX):
        outputs = AUX(times, x_values,y_values)
        px = outputs[:, 0] * torch.flatten(x_values) 
        vx = outputs[:, 1] * torch.flatten(x_values)
        ax = outputs[:, 2] * torch.flatten(x_values)

        py = outputs[:, 3] * torch.flatten(y_values)
        vy = outputs[:, 4] * torch.flatten(y_values)
        ay = outputs[:, 5] * torch.flatten(y_values)

        sxx = outputs[:, 6]
        sxy = outputs[:, 7]
        syy = outputs[:, 8]
        incompress=outputs[:,9]*0
        out=torch.stack( (px,vx,ax,py,vy,ay,sxx,sxy,syy,incompress),dim=1)
        return out
    def get_continuation_ouptuts(self,times,x_values,y_values,PINN,AUX):
        outputs = PINN(times, x_values,y_values)#*0 ################################################# Change needed ##############################
        aux_times=torch.zeros_like(times)
        aux_out=self.get_aux_ouptuts(aux_times,x_values,y_values,AUX)
        aux_out.grad=None

        # p = outputs[:, 0] * torch.flatten(x_values)*torch.flatten(times)+aux_p
        # v = outputs[:, 1] * torch.flatten(x_values)*torch.flatten(times)+aux_v
        # a = outputs[:, 2] * torch.flatten(x_values)*torch.flatten(times)+aux_a
        # s = outputs[:, 3]                          *torch.flatten(times)+aux_s


        px = outputs[:, 0] * torch.flatten(x_values) * torch.flatten(times)+aux_out[:,0]
        vx = outputs[:, 1] * torch.flatten(x_values) * torch.flatten(times)+aux_out[:,1]
        ax = outputs[:, 2] * torch.flatten(x_values) * torch.flatten(times)+aux_out[:,2]

        py = outputs[:, 3] * torch.flatten(y_values) * torch.flatten(times)+aux_out[:,3]
        vy = outputs[:, 4] * torch.flatten(y_values) * torch.flatten(times)+aux_out[:,4]
        ay = outputs[:, 5] * torch.flatten(y_values) * torch.flatten(times)+aux_out[:,5]

        sxx = outputs[:, 6] * torch.flatten(times)+aux_out[:,6]
        sxy = outputs[:, 7] * torch.flatten(times)+aux_out[:,7]
        syy = outputs[:, 8] * torch.flatten(times)+aux_out[:,8]
        incompress=outputs[:,9]*0


        out=torch.stack( (px,vx,ax,py,vy,ay,sxx,sxy,syy,incompress),dim=1)
        return out
    def get_loss(self, times, x_values, y_values, results):

        px = results[:, 0] 
        vx = results[:, 1] 
        ax = results[:, 2] 
        py = results[:, 3]
        vy = results[:, 4]
        ay = results[:, 5]
        sxx = results[:, 6]
        sxy = results[:, 7]
        syy = results[:, 8]
        incompress=torch.unsqueeze(torch.unsqueeze(results[:,9],1),1)
        ones=torch.ones_like(px)

        strain_x = torch.autograd.grad(px, x_values, ones, create_graph=True, retain_graph=True)[0]
        strain_y=torch.autograd.grad(py, y_values, ones, create_graph=True, retain_graph=True)[0]
        strain_xy=torch.autograd.grad(px, y_values, ones, create_graph=True, retain_graph=True)[0]
        strain_yx=torch.autograd.grad(py, x_values, ones, create_graph=True, retain_graph=True)[0]

        # x_components = torch.stack((strain_x, strain_xy), dim=1)
        # y_components = torch.stack((strain_yx, strain_y), dim=1)
        # F_old = torch.stack((x_components, y_components), dim=2)
        # F=F_old+self.eye 
        # volume=torch.det(F)
        # J=torch.unsqueeze(torch.unsqueeze(volume,1),1)
        # Ft = torch.transpose(F, 1, 2)
        # B = torch.matmul(F, Ft)
        # C = torch.matmul(Ft,F)
        # I1=C.diagonal(offset=0,dim1=-1,dim2=-2).sum(-1)
        # trace_b=B.diagonal(offset=0,dim1=-1,dim2=-2).sum(-1)
        # trace_b=torch.unsqueeze(torch.unsqueeze(trace_b,1),1)
        # #stress=2*self.k*B#-1e2*self.eye*(1-J)
        # #stress=2*self.k*B-self.eye*incompress
        # stress=2*self.k*B-1/3*self.eye*trace_b

        v_x = torch.autograd.grad(px, times, ones, create_graph=True, retain_graph=True)[0]
        a_x = torch.autograd.grad(vx, times, ones, create_graph=True, retain_graph=True)[0]
        v_y = torch.autograd.grad(py, times, ones, create_graph=True, retain_graph=True)[0]
        a_y = torch.autograd.grad(vy, times, ones, create_graph=True, retain_graph=True)[0]


        stress_gradientxx = torch.autograd.grad(sxx, x_values, ones, create_graph=True, retain_graph=True)[0]
        stress_gradientyy = torch.autograd.grad(sxx, y_values, ones, create_graph=True, retain_graph=True)[0]
        shear_gradientxy = torch.autograd.grad(sxy, x_values, ones, create_graph=True, retain_graph=True)[0]
        shear_gradientyx = torch.autograd.grad(sxy, y_values, ones, create_graph=True, retain_graph=True)[0]


        stress_DNNxx = self.k * torch.flatten(strain_x)
        stress_DNNyy = self.k * torch.flatten(strain_y)
        stress_DNNxy = self.k * torch.flatten(strain_xy)
        stress_DNNyx = self.k * torch.flatten(strain_yx)
        shear_DNN=.5*stress_DNNxy+.5*stress_DNNyx

        x_stresses=torch.stack((sxx,sxy),dim=1)
        y_stresses=torch.stack((sxy,syy),dim=1)
        DNN_stresses=torch.stack((x_stresses,y_stresses),dim=2)


        boundary_force_x=(torch.flatten(self.f+sxx)+(self.m*torch.flatten(ax)))*self.x_mask #TODO look at signs here
        boundary_force_y=(torch.flatten(self.f+syy)+(self.m*torch.flatten(ay)))*self.y_mask 
        boundary_loss=self.loss_function(boundary_force_x,torch.zeros_like(boundary_force_x))*1 + self.loss_function(boundary_force_y,torch.zeros_like(boundary_force_y))*1 #TODO assess if this boundary loss is a good solution to our problem of different physics at the boundary
        wave_equation_x=torch.flatten((-1*(torch.flatten(stress_gradientxx)+torch.flatten(shear_gradientxy)) + self.m * torch.flatten(ax)))*(1-self.x_mask) #we slice the loss becasue the wave equation does not hold when there is an external force
        wave_equation_y=torch.flatten((-1*(torch.flatten(stress_gradientyy)+torch.flatten(shear_gradientyx)) + self.m * torch.flatten(ay)))*(1-self.y_mask)
        
        wave_loss=self.loss_function(wave_equation_x,torch.zeros_like(wave_equation_x))*1+self.loss_function(wave_equation_y,torch.zeros_like(wave_equation_y))*1
        #stress_loss = self.loss_function(stress,DNN_stresses)
        stress_loss = self.loss_function(stress_DNNxx, sxx) * 1+self.loss_function(stress_DNNyy, syy) * 1+self.loss_function(shear_DNN, sxy) * 1
        v_loss = self.loss_function(vx, torch.flatten(v_x))*1+self.loss_function(vy, torch.flatten(v_y))
        a_loss = self.loss_function(ax, torch.flatten(a_x)) * 1+self.loss_function(ay, torch.flatten(a_y))
        volume=(1+strain_x)*(1+strain_y)-(strain_xy*strain_yx)
        

        #print(strain_x[0],strain_y[0],strain_xy[0],strain_yx[0])
        volume_goal=torch.ones_like(volume)
        volume_loss=self.loss_function(volume,volume_goal)*10
        #incompress_loss=self.loss_function(incompress,torch.zeros_like(incompress))*0.0

        total_loss = boundary_loss*1 + v_loss*1 + a_loss*1 + stress_loss+wave_loss*1+volume_loss*1#+energy_loss
        return total_loss, v_loss, a_loss, stress_loss, boundary_loss, wave_loss,volume_loss#,energy_loss
    def get_AUX_loss(self, times, x_values, y_values,results):
        px = results[:, 0] 
        vx = results[:, 1] 
        ax = results[:, 2] 
        py = results[:, 3]
        vy = results[:, 4]
        ay = results[:, 5]
        sxx = results[:, 6]
        sxy = results[:, 7]
        syy = results[:, 8]
        incompress=results[:,9]*0
        ones=torch.ones_like(px)

        strain_x = torch.autograd.grad(px, x_values, ones, create_graph=True, retain_graph=True)[0]
        strain_y=torch.autograd.grad(py, y_values, ones, create_graph=True, retain_graph=True)[0]
        strain_xy=torch.autograd.grad(px, y_values, ones, create_graph=True, retain_graph=True)[0]
        strain_yx=torch.autograd.grad(py, x_values, ones, create_graph=True, retain_graph=True)[0]

        # x_components = torch.stack((strain_x, strain_xy), dim=1)
        # y_components = torch.stack((strain_yx, strain_y), dim=1)
        # F_old = torch.stack((x_components, y_components), dim=2)
        # F=F_old+self.eye 
        # volume=torch.det(F)
        # J=torch.unsqueeze(torch.unsqueeze(volume,1),1)
        # Ft = torch.transpose(F, 1, 2)
        # B = torch.matmul(F, Ft)
        # C = torch.matmul(Ft,F)
        # I1=C.diagonal(offset=0,dim1=-1,dim2=-2).sum(-1)
        # trace_b=B.diagonal(offset=0,dim1=-1,dim2=-2).sum(-1)
        # trace_b=torch.unsqueeze(torch.unsqueeze(trace_b,1),1)
        # #stress=2*self.k*B#-1e2*self.eye*(1-J)
        # #stress=2*self.k*B-self.eye*incompress
        # stress=2*self.k*B-1/3*self.eye*trace_b


        stress_gradientxx = torch.autograd.grad(sxx, x_values, ones, create_graph=True, retain_graph=True)[0]
        stress_gradientyy = torch.autograd.grad(sxx, y_values, ones, create_graph=True, retain_graph=True)[0]
        shear_gradientxy = torch.autograd.grad(sxy, x_values, ones, create_graph=True, retain_graph=True)[0]
        shear_gradientyx = torch.autograd.grad(sxy, y_values, ones, create_graph=True, retain_graph=True)[0]


        stress_DNNxx = self.k * torch.flatten(strain_x)
        stress_DNNyy = self.k * torch.flatten(strain_y)
        stress_DNNxy = self.k * torch.flatten(strain_xy)
        stress_DNNyx = self.k * torch.flatten(strain_yx)
        shear_DNN=.5*stress_DNNxy+.5*stress_DNNyx

        x_stresses=torch.stack((sxx,sxy),dim=1)
        y_stresses=torch.stack((sxy,syy),dim=1)
        DNN_stresses=torch.stack((x_stresses,y_stresses),dim=2)


        boundary_force_x=(torch.flatten(self.f+sxx)+(self.m*torch.flatten(ax)))*self.x_mask #TODO look at signs here
        boundary_force_y=(torch.flatten(self.f+syy)+(self.m*torch.flatten(ay)))*self.y_mask 
        boundary_loss=self.loss_function(boundary_force_x,torch.zeros_like(boundary_force_x))*1 + self.loss_function(boundary_force_y,torch.zeros_like(boundary_force_y))*1 #TODO assess if this boundary loss is a good solution to our problem of different physics at the boundary.
        wave_equation_x=torch.flatten((-1*(torch.flatten(stress_gradientxx)+torch.flatten(shear_gradientxy)) + self.m * torch.flatten(ax)))*(1-self.x_mask) #we slice the loss becasue the wave equation does not hold when there is an external force
        wave_equation_y=torch.flatten((-1*(torch.flatten(stress_gradientyy)+torch.flatten(shear_gradientyx)) + self.m * torch.flatten(ay)))*(1-self.y_mask)
        wave_loss=self.loss_function(wave_equation_x,torch.zeros_like(wave_equation_x))*1+self.loss_function(wave_equation_y,torch.zeros_like(wave_equation_y))*1
        stress_loss = self.loss_function(stress_DNNxx, sxx) * 1+self.loss_function(stress_DNNyy, syy) * 1+self.loss_function(shear_DNN, sxy) * 1
        #stress_loss = self.loss_function(stress,DNN_stresses)
        volume=(1+strain_x)*(1+strain_y)-(strain_xy*strain_yx)

        #print(strain_x[0],strain_y[0],strain_xy[0],strain_yx[0])
        volume_goal=torch.ones_like(volume)
        volume_loss=self.loss_function(volume,volume_goal)

        kinetic=2*self.m*(torch.flatten(vx)*torch.flatten(vx)+torch.flatten(vy)*torch.flatten(vy))#TODO: Where did the 1/2 go?
        spring=.5*(torch.flatten(sxx)*torch.flatten(strain_x)+torch.flatten(syy)*torch.flatten(strain_y)+torch.flatten(sxy)*torch.flatten(strain_xy)+torch.flatten(sxy)*torch.flatten(strain_yx))
        total_energy=torch.mean(kinetic,0)+torch.mean(spring,0)
        inital_energy=torch.tensor((.self.k*(.5*.5+1/3*1/3))).to(device)
        #inital_energy=torch.tensor((self.k*(1.5**1.5+4/9-2))).to(device)
        #print(total_energy.data)
        #inital_energy=torch.tensor(.5*self.k).to(device)
        energy_loss=self.loss_function(total_energy,inital_energy)*10


        total_loss = boundary_loss + stress_loss+wave_loss+volume_loss+energy_loss
        return total_loss, stress_loss, boundary_loss, wave_loss,volume_loss,energy_loss

    def get_points(self,end,aux):
        if aux:
            x0_1=torch.zeros((self.offset0d*2+self.offset1d*1), requires_grad=True).to(device)
            x0_2=torch.zeros((self.offset1d*2+self.offset2d*1), requires_grad=True).to(device)
            x0_3=torch.zeros((self.offset0d*2+self.offset1d*1), requires_grad=True).to(device)
            x1_1=torch.ones((self.offset0d*2+self.offset1d*1), requires_grad=True).to(device)*self.l
            x1_2=torch.ones((self.offset1d*2+self.offset2d*1), requires_grad=True).to(device)*self.l
            x1_3=torch.ones((self.offset0d*2+self.offset1d*1), requires_grad=True).to(device)*self.l
            xrand_1=torch.rand((self.offset1d*2+self.offset2d*1), requires_grad=True).to(device)*self.l
            xrand_2=torch.rand((self.offset2d*2+self.offset3d*1), requires_grad=True).to(device)*self.l
            xrand_3=torch.rand((self.offset1d*2+self.offset2d*1), requires_grad=True).to(device)*self.l
            x_values=torch.cat((x0_1,xrand_1,x1_1,x0_2,xrand_2,x1_2,x0_3,xrand_3,x1_3))

            y0_0d=torch.zeros((self.offset0d), requires_grad=True).to(device)
            y0_1d=torch.zeros((self.offset1d), requires_grad=True).to(device)
            y0_2d=torch.zeros((self.offset2d), requires_grad=True).to(device)
            y1_0d=torch.ones((self.offset0d), requires_grad=True).to(device)*self.l
            y1_1d=torch.ones((self.offset1d), requires_grad=True).to(device)*self.l
            y1_2d=torch.ones((self.offset2d), requires_grad=True).to(device)*self.l
            yrand_1d=torch.rand((self.offset1d), requires_grad=True).to(device)*self.l
            yrand_2d=torch.rand((self.offset2d), requires_grad=True).to(device)*self.l
            yrand_3d=torch.rand((self.offset3d), requires_grad=True).to(device)*self.l
            y_values=torch.cat((y0_0d,yrand_1d,y1_0d,y0_1d,yrand_2d,y1_1d,y0_0d,yrand_1d,y1_0d,y0_1d,yrand_2d,y1_1d,y0_2d,yrand_3d,y1_2d,y0_1d,yrand_2d,y1_1d,y0_0d,yrand_1d,y1_0d,y0_1d,yrand_2d,y1_1d,y0_0d,yrand_1d,y1_0d))

            times=torch.ones_like(x_values,requires_grad=True)*end
            aux_times=torch.zeros_like(x_values,requires_grad=True)
            return times,aux_times,x_values,y_values

        else:
            t0=torch.zeros((self.offset0d*4+self.offset1d*4+self.offset2d*1), requires_grad=True).to(device)
            t_mid=torch.rand((self.offset1d*4+self.offset2d*4+self.offset3d*1), requires_grad=True).to(device)*end
            t1=torch.ones((self.offset0d*4+self.offset1d*4+self.offset2d*1), requires_grad=True).to(device)*end
            times=torch.cat((t0,t_mid,t1))

            x0_1=torch.zeros((self.offset0d*2+self.offset1d*1), requires_grad=True).to(device)
            x0_2=torch.zeros((self.offset1d*2+self.offset2d*1), requires_grad=True).to(device)
            x0_3=torch.zeros((self.offset0d*2+self.offset1d*1), requires_grad=True).to(device)
            x1_1=torch.ones((self.offset0d*2+self.offset1d*1), requires_grad=True).to(device)*self.l
            x1_2=torch.ones((self.offset1d*2+self.offset2d*1), requires_grad=True).to(device)*self.l
            x1_3=torch.ones((self.offset0d*2+self.offset1d*1), requires_grad=True).to(device)*self.l
            xrand_1=torch.rand((self.offset1d*2+self.offset2d*1), requires_grad=True).to(device)*self.l
            xrand_2=torch.rand((self.offset2d*2+self.offset3d*1), requires_grad=True).to(device)*self.l
            xrand_3=torch.rand((self.offset1d*2+self.offset2d*1), requires_grad=True).to(device)*self.l
            x_values=torch.cat((x0_1,xrand_1,x1_1,x0_2,xrand_2,x1_2,x0_3,xrand_3,x1_3))

            y0_0d=torch.zeros((self.offset0d), requires_grad=True).to(device)
            y0_1d=torch.zeros((self.offset1d), requires_grad=True).to(device)
            y0_2d=torch.zeros((self.offset2d), requires_grad=True).to(device)
            y1_0d=torch.ones((self.offset0d), requires_grad=True).to(device)*self.l
            y1_1d=torch.ones((self.offset1d), requires_grad=True).to(device)*self.l
            y1_2d=torch.ones((self.offset2d), requires_grad=True).to(device)*self.l
            yrand_1d=torch.rand((self.offset1d), requires_grad=True).to(device)*self.l
            yrand_2d=torch.rand((self.offset2d), requires_grad=True).to(device)*self.l
            yrand_3d=torch.rand((self.offset3d), requires_grad=True).to(device)*self.l
            y_values=torch.cat((y0_0d,yrand_1d,y1_0d,y0_1d,yrand_2d,y1_1d,y0_0d,yrand_1d,y1_0d,y0_1d,yrand_2d,y1_1d,y0_2d,yrand_3d,y1_2d,y0_1d,yrand_2d,y1_1d,y0_0d,yrand_1d,y1_0d,y0_1d,yrand_2d,y1_1d,y0_0d,yrand_1d,y1_0d))

            return times,x_values,y_values
    def plot(self,PINN):
        num_points=101
        x=torch.linspace(0,self.l,num_points,requires_grad=True).to(device)
        t=torch.linspace(0, 1, num_points, requires_grad=True).to(device)
        gridx,gridt=torch.meshgrid(x,t,indexing='ij')
        p,v,a,s,m=self.get_ouptuts(gridt,gridx,PINN)
        fig = plt.figure()
        ax1 = fig.add_subplot(projection='3d')
        ax1.scatter(gridx.data.cpu().detach().numpy(),gridt.data.cpu().detach().numpy(),p.data.cpu().detach().numpy())
        ax1.set_xlabel('X ')
        ax1.set_ylabel('Time')
        ax1.set_zlabel('Position')
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
            models.append(mpinn)
            torch.save(mpinn.state_dict(), "model"+str(i+1)+".pt")
            if i<num_models-2:
                mAux=PINN.train_AUX(mpinn,interval,old_aux=aux[-1])
                aux.append(mAux)
                torch.save(mAux.state_dict(), "aux"+str(i+1)+".pt")


m=1
k=1
l=1
f=0
PINN=PINN(m,k,l,f)
run=False
aux=False
mp1=False
aux2=False
mp2=False

t0=time.time()


# pin_start=PINN_Net()
# pin_start.load_state_dict(torch.load("start_model.pt"))
# time=torch.tensor([1.])
# x=torch.tensor([.5])
# y=torch.tensor([1/3])
# out=PINN.get_ouptuts(time,x,y,pin_start)
# vx1=out[:,4]
# print(vx1)
# aux_start=Aux_Net()
# aux_start.load_state_dict(torch.load("start_aux.pt"))
# time=torch.tensor([0])
# x=torch.tensor([.5])
# y=torch.tensor([1/3])
# out=PINN.get_aux_ouptuts(time,x,y,aux_start)
# vx=out[:,4]
# print(vx)
# print(torch.square(vx1-vx))
# quit()


PINN.full_run(num_models=16,end_time=16)

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

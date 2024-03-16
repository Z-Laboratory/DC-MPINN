import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import imageio
import os
import csv

scale=1
scale2=1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np



class PDE_Net(nn.Module):
    def __init__(self):
        super(PDE_Net, self).__init__()
        a = 100
        self.hidden_layer1 = nn.Linear(4, a)
        self.hidden_layer2 = nn.Linear(a, a)
        self.hidden_layer3 = nn.Linear(a, a)
        self.hidden_layer4 = nn.Linear(a, a)
        self.hidden_layer5 = nn.Linear(a, a)
        self.hidden_layer6 = nn.Linear(a, a)
        self.hidden_layer7 = nn.Linear(a, a)
        self.hidden_layer8 = nn.Linear(a, a)
        self.output_layer = nn.Linear(a, 10)
        #self.activation = torch.relu
        self.activation=torch.nn.LeakyReLU()
        '''
        relu is much more erratic, but produces better absolute results faster
        tanh and sigmoid give a stable lower bound, but are slower
        '''

    def forward(self,t, x, y, z):
        inputs = torch.cat([t,x, y, z], axis=1)  # combined two arrays of 1 columns each to one array of 2 columns
        layer1_out = self.activation(self.hidden_layer1(inputs))
        layer2_out = self.activation(self.hidden_layer2(layer1_out))
        layer3_out = self.activation(self.hidden_layer3(layer2_out)+layer1_out)
        layer4_out = self.activation(self.hidden_layer4(layer3_out))
        layer5_out = self.activation(self.hidden_layer5(layer4_out)+layer3_out)
        layer6_out = self.activation(self.hidden_layer6(layer5_out))
        layer7_out = self.activation(self.hidden_layer7(layer6_out)+layer5_out)
        layer8_out = self.activation(self.hidden_layer8(layer7_out))
        output = self.output_layer(layer3_out)  ## For regression, no activation is used in output layer


        zeros=torch.zeros_like(x)
        ones=torch.ones_like(x)*1/3
        #output[:, 0] = torch.flatten(zeros) #disp_x
        #output[:, 1] = torch.flatten(x) #disp_y
        #output[:, 1] = torch.flatten(zeros) #disp_z
        #output[:, 3] = torch.flatten(zeros) #stress_x
        #output[:, 4] = torch.flatten(zeros) #stress_y
        #output[:, 5] = torch.flatten(zeros) #stress_z
        #output[:, 6] = torch.flatten(ones)  # shear_xy
        #output[:, 7] = torch.flatten(ones)  # shear_xz
        #output[:, 8] = torch.flatten(zeros)  # shear_yz
        #output[:, 9] = torch.flatten(zeros)  #p


        return output


class PDE_Net2(nn.Module):
    def __init__(self):
        super(PDE_Net2, self).__init__()
        a = 200
        self.hidden_layer1 = nn.Linear(4, a)
        self.hidden_layer2 = nn.Linear(a, a)
        self.hidden_layer3 = nn.Linear(a, a)
        self.hidden_layer4 = nn.Linear(a, a)
        self.hidden_layer5 = nn.Linear(a, a)
        self.hidden_layer6 = nn.Linear(a, a)
        self.hidden_layer7 = nn.Linear(a, a)
        self.hidden_layer8 = nn.Linear(a, a)
        self.output_layer = nn.Linear(a, 10)
        #self.activation = torch.relu
        self.activation=torch.nn.LeakyReLU()
        '''
        relu is much more erratic, but produces better absolute results faster
        tanh and sigmoid give a stable lower bound, but are slower
        '''

    def forward(self,t, x, y, z):
        inputs = torch.cat([t,x, y, z], axis=1)  # combined two arrays of 1 columns each to one array of 2 columns
        layer1_out = self.activation(self.hidden_layer1(inputs))
        layer2_out = self.activation(self.hidden_layer2(layer1_out))
        layer3_out = self.activation(self.hidden_layer3(layer2_out))
        # layer4_out = self.activation(self.hidden_layer4(layer3_out))
        # layer5_out = self.activation(self.hidden_layer5(layer4_out))
        # layer6_out = self.activation(self.hidden_layer6(layer5_out))
        # layer7_out = self.activation(self.hidden_layer7(layer6_out))
        # layer8_out = self.activation(self.hidden_layer8(layer7_out))
        output = self.output_layer(layer3_out)  ## For regression, no activation is used in output layer


        zeros=torch.zeros_like(x)
        ones=torch.ones_like(x)*1/3
        #output[:, 0] = torch.flatten(zeros) #disp_x
        #output[:, 1] = torch.flatten(x) #disp_y
        #output[:, 1] = torch.flatten(zeros) #disp_z
        #output[:, 3] = torch.flatten(zeros) #stress_x
        #output[:, 4] = torch.flatten(zeros) #stress_y
        #output[:, 5] = torch.flatten(zeros) #stress_z
        #output[:, 6] = torch.flatten(ones)  # shear_xy
        #output[:, 7] = torch.flatten(ones)  # shear_xz
        #output[:, 8] = torch.flatten(zeros)  # shear_yz
        #output[:, 9] = torch.flatten(zeros)  #p


        return output


def dist(t,x,y,z):
    a = (y) #* (y - 1) * (z - 1)*x*y*z
    b=torch.ones_like(a)

    all_edges=(1-x)*(2-y)*(1-z)*x*y*z
    all_2=(1-x)*(1-z)*x*z
    distance_function = (t)*torch.cat((y*x, y ,y*z,(1-x), 1-y, 1-z, 1-y, b,1-y,b), dim=1)
    return distance_function

def f(s1, x, y, z, net, epoch,points):


    distance_function=dist(s1,x,y,z)

    inital = torch.zeros((points,10)).to(device)
    inital[:,4]=torch.flatten(s1)
    outputs_BC=inital


    output_PDE = distance_function * net(s1,x, y, z)  # output is the 3 displacements and 3 principal stresses
    output = outputs_BC + output_PDE


    disp_x = output[:, 0]  # *0+torch.flatten(x)
    disp_y = output[:, 1]  # *0
    disp_z = output[:, 2]  # *0
    stress_x = output[:, 3]  # *0+1
    stress_y = output[:, 4]  # *0+0
    stress_z = output[:, 5]  # *0+0
    shear_xy = output[:, 6]  # *0+1
    shear_xz = output[:, 7]  # *0+0
    shear_yz = output[:, 8]  # *0+0
    p = torch.reshape(output[:, 9], (points, 1, 1))


    ones=torch.ones_like(x)
    ones=torch.flatten(ones)
    #ones=None

    #Divergence variables

    #stress divergence
    stress_divergence_x = torch.autograd.grad(stress_x, x,grad_outputs=ones, create_graph=True)[0]
    stress_divergence_y = torch.autograd.grad(stress_y, y,grad_outputs=ones, create_graph=True)[0]
    stress_divergence_z = torch.autograd.grad(stress_z, z,grad_outputs=ones, create_graph=True)[0]
    #shear divergence
    shear_divergence_xy = torch.autograd.grad(shear_xy, y,grad_outputs=ones, create_graph=True)[0]
    shear_divergence_xz = torch.autograd.grad(shear_xz, z,grad_outputs=ones, create_graph=True)[0]
    shear_divergence_yz = torch.autograd.grad(shear_yz, z,grad_outputs=ones, create_graph=True)[0]
    shear_divergence_yx = torch.autograd.grad(shear_xy, x,grad_outputs=ones, create_graph=True)[0]
    shear_divergence_zx = torch.autograd.grad(shear_xz, x,grad_outputs=ones, create_graph=True)[0]
    shear_divergence_zy = torch.autograd.grad(shear_yz, y,grad_outputs=ones, create_graph=True)[0]


    #this is proper implementation of cauchy momentum equation
    div_x=stress_divergence_x+shear_divergence_xy+shear_divergence_xz
    div_y=stress_divergence_y+shear_divergence_yz+shear_divergence_yx
    div_z=stress_divergence_z+shear_divergence_zx+shear_divergence_zy

    #strain variables
    strain_x = torch.autograd.grad(disp_x, x,grad_outputs=ones, create_graph=True)[0]
    strain_y = torch.autograd.grad(disp_y, y,grad_outputs=ones, create_graph=True)[0]
    strain_z = torch.autograd.grad(disp_z, z,grad_outputs=ones, create_graph=True)[0]
    strain_xy = torch.autograd.grad(disp_x, y,grad_outputs=ones, create_graph=True)[0]
    strain_xz = torch.autograd.grad(disp_x, z,grad_outputs=ones, create_graph=True)[0]
    strain_yz = torch.autograd.grad(disp_y, z,grad_outputs=ones, create_graph=True)[0]
    strain_yx = torch.autograd.grad(disp_y, x,grad_outputs=ones, create_graph=True)[0]
    strain_zx = torch.autograd.grad(disp_z, x,grad_outputs=ones, create_graph=True)[0]
    strain_zy = torch.autograd.grad(disp_z, y,grad_outputs=ones, create_graph=True)[0]



    eye1 = [[[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]]
    eye1 = torch.tensor(eye1, device=device)
    eye = eye1.repeat(points, 1, 1)

    incompress = eye * p

    x_components = torch.cat((strain_x, strain_xy, strain_xz), dim=1)
    y_components = torch.cat((strain_yx, strain_y, strain_yz), dim=1)
    z_components = torch.cat((strain_zx, strain_zy, strain_z), dim=1)
    F = torch.stack((x_components, y_components, z_components), dim=2)

    F=F+eye




    volume=torch.linalg.det(F)
    volume_ones = torch.ones_like(volume)



    Ft = torch.transpose(F, 1, 2)

    B = torch.matmul(F, Ft)
    stress = 2*.166 * B - eye*p

    stress_divergence = torch.cat((div_x, div_y, div_z))


    divergence_zeros = torch.zeros_like(stress_divergence)


    stress_x = torch.reshape(stress_x, (points, 1))
    stress_y = torch.reshape(stress_y, (points, 1))
    stress_z = torch.reshape(stress_z, (points, 1))
    shear_xy = torch.reshape(shear_xy, (points, 1))
    shear_xz = torch.reshape(shear_xz, (points, 1))
    shear_yz = torch.reshape(shear_yz, (points, 1))

    x_comp = torch.cat((stress_x, shear_xy, shear_xz), dim=1)
    y_comp = torch.cat((shear_xy, stress_y, shear_yz), dim=1)
    z_comp = torch.cat((shear_xz, shear_yz, stress_z), dim=1)

    stress_DNN = torch.stack((x_comp, y_comp, z_comp), dim=2)




    mse_cost_function = torch.nn.MSELoss()

    stress_divergence_loss = mse_cost_function(divergence_zeros, stress_divergence)

    linear_elasticity_loss = mse_cost_function(stress, stress_DNN)

    volume_loss=mse_cost_function(volume,volume_ones)

    if epoch == 499:
        a = 0
        np_B = B.cpu().detach().numpy()
        np_incompress=incompress.cpu().detach().numpy()
        np_F = F.cpu().detach().numpy()
        np_volume = volume.cpu().detach().numpy()
        np_outputs = output.cpu().detach().numpy()
        np_stress = stress.cpu().detach().numpy()
        np_stressDNN = stress_DNN.cpu().detach().numpy()
        np_strain_x=strain_x.cpu().detach().numpy()
        np_strain_y = strain_y.cpu().detach().numpy()
        np_strain_z = strain_z.cpu().detach().numpy()
        np_p = p.cpu().detach().numpy()


        # stress divergence
        np_stress_divergence_x=stress_divergence_x.cpu().detach().numpy()
        np_stress_divergence_y=stress_divergence_y.cpu().detach().numpy()
        np_stress_divergence_z=stress_divergence_z.cpu().detach().numpy()
        # shear divergence
        np_shear_divergence_xy=shear_divergence_xy.cpu().detach().numpy()
        np_shear_divergence_xz=shear_divergence_xz.cpu().detach().numpy()
        np_shear_divergence_yz=shear_divergence_yz.cpu().detach().numpy()
        np_shear_divergence_yx=shear_divergence_yx.cpu().detach().numpy()
        np_shear_divergence_zx=shear_divergence_zx.cpu().detach().numpy()
        np_shear_divergence_zy=shear_divergence_zy.cpu().detach().numpy()

        np_div_x=div_x.cpu().detach().numpy()
        np_div_y=div_y.cpu().detach().numpy()
        np_div_z=div_z.cpu().detach().numpy()
        a = 0

    loss = [stress_divergence_loss, linear_elasticity_loss,volume_loss]

    return loss

def BC_F(s1,x, y, z, flag, net):
    stress1 = torch.reshape(s1, (-1,))

    output = net(s1,x, y, z)  # output is the 3 displacements and 3 principal stresses


    disp_x = output[:, 0]  # *0+torch.flatten(x)
    disp_y = output[:, 1]  # *0
    disp_z = output[:, 2]  # *0
    stress_x = output[:, 3]  # *0+1
    stress_y = output[:, 4]  # *0+0
    stress_z = output[:, 5]  # *0+0
    shear_xy = output[:, 6]  # *0+1
    shear_xz = output[:, 7]  # *0+0
    shear_yz = output[:, 8]  # *0+0
    all_zeros = torch.zeros_like(disp_x)
    all_ones = torch.ones_like(stress_x)

    mse_cost_function = torch.nn.MSELoss()

    # flag indicates that we are doing the first BC: a 0 displacement condition on the bottom surface
    if flag == 1:
        BC_loss_disp = mse_cost_function(disp_x, all_zeros)
        BC_loss_disp += mse_cost_function(disp_y, all_zeros)
        BC_loss_disp += mse_cost_function(disp_z, all_zeros)
        return BC_loss_disp

    elif flag == 2:
        BC_loss_stress = mse_cost_function(stress_y, stress1)
        BC_loss_stress += mse_cost_function(shear_xy, all_zeros)
        BC_loss_stress += mse_cost_function(shear_yz, all_zeros)
        return BC_loss_stress


    elif flag == 3:
        BC_loss_stress = mse_cost_function(stress_x, all_zeros)
        return BC_loss_stress

    elif flag == 4:
        BC_loss_disp = mse_cost_function(disp_x, all_zeros)
        return BC_loss_disp

    elif flag == 5:
        BC_loss_stress = mse_cost_function(stress_z, all_zeros)
        return BC_loss_stress

    elif flag == 6:
        BC_loss_disp = mse_cost_function(disp_z, all_zeros)
        return BC_loss_disp

    elif flag ==7:
        BC_loss_disp = mse_cost_function(disp_x, all_zeros)
        BC_loss_disp += mse_cost_function(disp_y, all_zeros)
        BC_loss_disp += mse_cost_function(disp_z, all_zeros)
        return BC_loss_disp

def train_PDE(net, optimizer,optimizer_version):  # ,mse_cost_function,x_bc,t_bc,u_bc):
    ### (3) Training / Fitting
    if optimizer_version == 2:
        iterations = 501
    if optimizer_version == 1:
        iterations = 1
    losses = np.zeros(iterations)
    length=1



    for epoch in range(iterations):
        optimizer.zero_grad()  # to make the gradients zero

        # set up "mesh" variables
        num_collocation = 1500
        x = np.random.uniform(0, 1, num_collocation)
        y = np.random.uniform(0, 1, num_collocation)
        z = np.random.uniform(0, 1, num_collocation)

        stress1 = np.random.uniform(0, 1, num_collocation) * scale

        pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
        pt_y = Variable(torch.from_numpy(y).float(), requires_grad=True).to(device)
        pt_z = Variable(torch.from_numpy(z).float(), requires_grad=True).to(device)
        stress1 = Variable(torch.from_numpy(stress1).float(), requires_grad=True).to(device)
        x = torch.reshape(pt_x, (num_collocation, 1))
        y = torch.reshape(pt_y, (num_collocation, 1))
        z = torch.reshape(pt_z, (num_collocation, 1))
        stress1 = torch.reshape(stress1, (num_collocation, 1))


        # Combining the loss functions
        if optimizer_version==1:
            def closure():
                stress_divergence_loss, linear_elasticity_loss, volume_loss = f(stress1, x, y, z, net, epoch, num_collocation)  # output of f(x,t)
                loss = stress_divergence_loss + linear_elasticity_loss+volume_loss
                loss.backward()  # This is for computing gradients using backward propagation
                with torch.autograd.no_grad():
                    if epoch % 100 == 0:
                        print(epoch, "Traning Loss:", loss.data)
                        print(epoch, "Traning Loss:", stress_divergence_loss.data, linear_elasticity_loss.data,
                              volume_loss.data)
                    losses[epoch] = loss.data
                return loss
            optimizer.step(closure)  # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

        if optimizer_version==2:
            stress_divergence_loss, linear_elasticity_loss, volume_loss = f(stress1, x, y, z, net, epoch, num_collocation)  # output of f(x,t)
            loss = stress_divergence_loss + linear_elasticity_loss+volume_loss
            loss.backward()  # This is for computing gradients using backward propagation
            optimizer.step() # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta
            with torch.autograd.no_grad():
                if epoch % 100 == 0:
                    print(epoch, "Traning Loss:", loss.data)
                    print(epoch, "Traning Loss:", stress_divergence_loss.data, linear_elasticity_loss.data,
                          volume_loss.data)
                losses[epoch] = loss.data



    return losses


if __name__ == "__main__":

    time_to_evaluate=0

    direction1=1
    direction2=0

    a=False
    single_eval=a
    trainBC=a
    trainPDE=a
    make_plots=True

    plot_surf=False

    random_runs=False
    setup = True
    evaluate = True

    if time_to_evaluate>1:
        time_to_evaluate=.99

    with open('different_BC_PINN_noBCnet.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["direction1", "direction2", "np.max(disp_x + x)",
                         "np.max(disp_y + y)", "np.max(disp_z + z)"])
        while time_to_evaluate<=1.001:



            if setup:
                start = time.perf_counter()
                net = PDE_Net2()
                net = net.to(device)
                optimizer = torch.optim.Rprop(net.parameters(),lr=.0002,step_sizes=[1e-8,1])
                optimizer2 = torch.optim.LBFGS(net.parameters(),lr=.05, history_size=150, max_iter=15,line_search_fn='strong_wolfe')


                E = torch.ones(1, device=device) * 1
                E.to(device)


                if trainPDE:
                    losses = train_PDE(net, optimizer, optimizer_version=2)
                    torch.save(net.state_dict(), "PDE_extension.pt")
                    plt.plot(losses)
                    # plt.yscale('log')
                    #losses = train_PDE(BC_net, net, optimizer2, optimizer_version=1)
                    torch.save(net.state_dict(), "PDE_extension.pt")
                    plt.plot(losses)
                    plt.yscale('log')

                net.load_state_dict(torch.load("PDE_extension.pt"))

                end = time.perf_counter()

                print(f"Trained model in {end - start:0.4f} seconds")
                plt.show()


            if evaluate:
                x = np.random.uniform(0, 1, 398)
                x = np.append(x, [.5, .5])
                y = np.random.uniform(0, .95, 398)
                y = np.append(y, [0, 1])
                z = np.random.uniform(0, 1, 398)
                z = np.append(z, [.5, .5])

                if plot_surf:
                    x = np.array([0,0,0,0,1,1,1,1])
                    y = np.array([0,0,1,1,0,0,1,1])
                    z = np.array([0,1,0,1,0,1,0,1])

                pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
                pt_y = Variable(torch.from_numpy(y).float(), requires_grad=True).to(device)
                pt_z = Variable(torch.from_numpy(z).float(), requires_grad=True).to(device)
                pt_x_collocation = torch.reshape(pt_x, (np.size(x), 1))
                pt_y_collocation = torch.reshape(pt_y, (np.size(x), 1))
                pt_z_collocation = torch.reshape(pt_z, (np.size(x), 1))



                #t = torch.ones_like(pt_x_collocation) * direction1 * scale*time_to_evaluate
                t=torch.ones_like(pt_x_collocation)*time_to_evaluate

                distance_function =dist(t,pt_x_collocation,pt_y_collocation,pt_z_collocation)

                inital = torch.zeros((400, 10))
                inital[:, 4] = torch.ones(400)
                outputs_BC = inital.to(device)

                outputs_PDE = distance_function * net(t,pt_x_collocation, pt_y_collocation, pt_z_collocation)
                outputs = outputs_PDE + outputs_BC
                #outputs=net(t, pt_x_collocation, pt_y_collocation, pt_z_collocation)*distance_function
                # outputs=net(pt_x_collocation,pt_y_collocation,pt_z_collocation)

                disp_x = outputs[:, 0]
                disp_y = outputs[:, 1]
                disp_z = outputs[:, 2]

                stress_x = outputs[:, 3]
                stress_y = outputs[:, 4]
                stress_z = outputs[:, 5]

                shear_xy = outputs[:, 6]  # *0+1
                shear_xz = outputs[:, 7]  # *0+0
                shear_yz = outputs[:, 8]  # *0+0

                # a=torch.mean(outputs,dim=1)

                stress_divergence_x = torch.autograd.grad(stress_x.sum(), pt_x_collocation, create_graph=True)[0]
                stress_divergence_y = torch.autograd.grad(stress_y.sum(), pt_y_collocation, create_graph=True)[0]
                stress_divergence_z = torch.autograd.grad(stress_z.sum(), pt_z_collocation, create_graph=True)[0]
                strain_x = torch.autograd.grad(disp_x.sum(), pt_x_collocation, create_graph=True)[0]
                strain_y = torch.autograd.grad(disp_y.sum(), pt_y_collocation, create_graph=True)[0]
                strain_z = torch.autograd.grad(disp_z.sum(), pt_z_collocation, create_graph=True)[0]
                strain_xy = torch.autograd.grad(disp_x.sum(), pt_y_collocation, create_graph=True)[0]
                strain_xz = torch.autograd.grad(disp_x.sum(), pt_z_collocation, create_graph=True)[0]
                strain_yz = torch.autograd.grad(disp_y.sum(), pt_z_collocation, create_graph=True)[0]
                strain_yx = torch.autograd.grad(disp_y.sum(), pt_x_collocation, create_graph=True)[0]
                strain_zx = torch.autograd.grad(disp_z.sum(), pt_x_collocation, create_graph=True)[0]
                strain_zy = torch.autograd.grad(disp_z.sum(), pt_y_collocation, create_graph=True)[0]
                shear_divergence_xy = torch.autograd.grad(shear_xy.sum(), pt_y_collocation, create_graph=True)[0]
                shear_divergence_xz = torch.autograd.grad(shear_xz.sum(), pt_z_collocation, create_graph=True)[0]
                shear_divergence_yz = torch.autograd.grad(shear_yz.sum(), pt_z_collocation, create_graph=True)[0]

                strain_x = strain_x.cpu().detach().numpy()
                strain_y = strain_y.cpu().detach().numpy()
                strain_z = strain_z.cpu().detach().numpy()
                strain_xy = strain_xy.cpu().detach().numpy()
                strain_xz = strain_xz.cpu().detach().numpy()
                strain_yz = strain_yz.cpu().detach().numpy()
                strain_yx = strain_yx.cpu().detach().numpy()
                strain_zx = strain_zx.cpu().detach().numpy()
                strain_zy = strain_zy.cpu().detach().numpy()
                AAoutputs = outputs.data.cpu().detach().numpy()
                AABCoutputs = outputs_BC.data.cpu().detach().numpy()
                stress_x = AAoutputs[:, 3]
                stress_y = AAoutputs[:, 4]
                stress_z = AAoutputs[:, 5]
                disp_x = AAoutputs[:, 0]
                disp_y = AAoutputs[:, 1]
                disp_z = AAoutputs[:, 2]
                shear_xy = AAoutputs[:, 6]  # *0+1
                shear_xz = AAoutputs[:, 7]  # *0+0
                shear_yz = AAoutputs[:, 8]  # *0+0


                # it seems that the displacements in the y and z directions are not stable


                # print(np.max(disp_x + x))
                # print(np.max(disp_y + y))
                # print(np.max(disp_z + z))



            lam=np.max(disp_x + x)/np.max(x)
            vol=np.max(disp_x + x) * np.max(disp_y + y) * np.max(disp_z + z)
            nh_stress=1/3*(lam*lam-1/(lam))
            # print("current volume is: ",vol," Original is 2")
            # print("NH Stress needed for this deformation ",nh_stress," Actual stress is ",time_to_evaluate)
            #
            print("max_x", np.max(disp_x + x))
            print("max_y", np.max(disp_y + y))
            print("max_z", np.max(disp_z + z))

            writer.writerow([time_to_evaluate, direction2, np.max(disp_x + x),
                             np.max(disp_y + y), np.max(disp_z + z)])


            time_to_evaluate+=.01
            # if direction1>=1:
            #     if direction2>=1:
            #         time_to_evaluate=1.01
            #     else:
            #         direction2+=.05
            #         direction1=0
            #
            # else:
            #     direction1+=.05

            # print(np.max(disp_x + x)*np.max(disp_y + y)*np.max(disp_z + z))
            if make_plots == True:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.scatter(disp_x + x, disp_y + y, disp_z + z, 'g')
                ax.scatter(x, y, z, 'b')
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_xlim(-1, 2)
                ax.set_ylim(-1, 2)
                ax.set_zlim(-1, 2)
                ax.view_init(elev=-90., azim=90)

                if int(time_to_evaluate*100)<10:
                    a="0"
                else:
                    a=""
                #plt.savefig('Bone/' +a+ str(int(time_to_evaluate*100)) + '.png')

            if plot_surf:

                def midpoints(x):
                    sl = ()
                    for i in range(x.ndim):
                        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
                        sl += np.index_exp[:]
                    return x

                x, y, z = np.mgrid[0:1:2j, 0:1:2j, 0:1:2j]

                update_x = disp_x.reshape((2, 2, 2))
                update_y = disp_y.reshape((2, 2, 2))
                update_z = disp_z.reshape((2, 2, 2))
                x += update_x
                y += update_y
                z += update_z

                shading = midpoints(x)

                fill = np.ones_like(shading)

                ax = plt.figure().add_subplot(projection='3d')
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_xlim(0, 2)
                ax.set_ylim(0, 2)
                ax.set_zlim(0, 1)
                ax.view_init(elev=45., azim=45)
                ax.voxels(x, y, z, fill)




                if int(time_to_evaluate*100)<10:
                    a="0"
                else:
                    a=""
                #plt.savefig('1d_beam/' +a+ str(int(time_to_evaluate*100)) + '.png')

            #plt.show()
            #time_to_evaluate += 1 / 100

            if single_eval == True:
                plt.show()
                quit()
            plt.close()

    #
    # # with imageio.get_writer('mygif.gif', mode='I') as writer:
    # images=[]
    # for filename in os.listdir('2d_plate'):
    #     image = imageio.imread('2d_plate/'+filename)
    #     images.append(image)
    # imageio.mimsave('./2d_plate.gif', # output gif
    #                 images,          # array of input frames
    #                 fps = 10)         # optional: frames per second
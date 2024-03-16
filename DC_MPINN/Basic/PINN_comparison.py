import torch
import torch.nn as nn
from torch.autograd import Variable
import time
from scipy.optimize import fsolve
import csv
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PDE_Net(nn.Module):
    def __init__(self):
        super(PDE_Net, self).__init__()
        a = 500 #increasing this is a great way to inprove accuracy (at the cost of compute time)
        self.hidden_layer1 = nn.Linear(6, a)
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

    def forward(self,s1,s2,s3, x, y, z):
        inputs = torch.cat([s1,s2,s3,x, y, z], axis=1)  # combined two arrays of 1 columns each to one array of 2 columns
        layer1_out = self.activation(self.hidden_layer1(inputs))
        layer2_out = self.activation(self.hidden_layer2(layer1_out))
        # layer3_out = self.activation(self.hidden_layer3(layer2_out))
        # layer4_out = self.activation(self.hidden_layer4(layer3_out))
        # layer5_out = self.activation(self.hidden_layer5(layer4_out))
        # layer6_out = self.activation(self.hidden_layer6(layer5_out))
        # layer7_out = self.activation(self.hidden_layer7(layer6_out))
        # layer8_out = self.activation(self.hidden_layer8(layer7_out))
        output = self.output_layer(layer2_out)  ## For regression, no activation is used in output layer
        return output

def analytic_equations(p, *stresses):
    c1 = .166 * 2
    s1, s2, s3 = stresses
    l1, l2, l3 = p
    # s1 - s2 = c1 * (l1 ** 2 - l2 ** 2)
    # s1 - s2 = c1 * (l1 ** 2 - l3 ** 2)
    # s2 - s3 = c1 * (l2 ** 2 - l3 ** 2)
    # L1 * l2 * l3 = 1
    return (c1 * (l1 ** 2 - l2 ** 2) - s1 + s2, c1 * (l1 ** 2 - l3 ** 2) - s1 + s3, l1 * l2 * l3 - 1)

def dist(s1,s2,s3,x,y,z):
    a = (y)
    b=torch.ones_like(a)
    c=torch.zeros_like(a)
    distance_function =torch.cat((x, y+1 ,z,1-x, 1-y, 1-z, c, c,c,b), dim=1)
    return distance_function

def get_BC(s1,s2,s3,x,y,z):
    a = (y)
    b=torch.ones_like(a)
    c=torch.zeros_like(a)
    BC = torch.cat((x, y+1 ,z,s1, s2, s3, c, c,c,b), dim=1)
    return BC

def f(t,s1,s2,s3, x, y, z, net, epoch,points):


    distance_function=dist(s1,s2,s3,x,y,z)

    outputs_BC=get_BC(s1,s2,s3,x,y,z)


    output_PDE = distance_function * net(s1,s2,s3,x, y, z)  # output is the 3 displacements and 3 principal stresses
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
    I1=B[:,0,0]+B[:,1,1]+B[:,2,2]
    I=eye

    C1=.166
    D1=1000
    #incompressible
    #stress = 2*C1 * B - eye*p
    #compressible (Rivilin)
    p=-2*D1*(volume*(volume-volume_ones))
    J_pow=torch.pow(volume,-2/3)
    Ibar=I1*J_pow
    Bbar=torch.mul(J_pow,B)
    dev_Bbar=Bbar-1/3*Ibar*I
    stress=-p*I+2*C1*dev_Bbar



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

    if epoch == 199:
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

def train_PDE(net, optimizer,optimizer_version):  # ,mse_cost_function,x_bc,t_bc,u_bc):
    ### (3) Training / Fitting
    if optimizer_version == 2:
        iterations = 201
    if optimizer_version == 1:
        iterations = 1
    losses = np.zeros(iterations)
    length=1



    for epoch in range(iterations):
        optimizer.zero_grad()  # to make the gradients zero

        # set up "mesh" variables
        num_collocation = 5000
        x = np.random.uniform(-0.1, 1.1, num_collocation)
        y = np.random.uniform(-1.1, 1.1, num_collocation)
        z = np.random.uniform(-0.1, 1.1, num_collocation)
        # x = np.random.choice(np.array([0,.5,1]),num_collocation)
        # y = np.random.choice(np.array([0,.5,1]),num_collocation)
        # z = np.random.choice(np.array([0,.5,1]),num_collocation)

        #time = np.random.uniform(-0.1, 1.1, num_collocation)
        time=np.ones_like(x)
        stress1 = np.random.uniform(-0.1, 1.1, num_collocation)
        stress2 = np.random.uniform(-0.1, 1.1, num_collocation)
        stress3 = np.random.uniform(-0.1, 1.1, num_collocation)

        #if we want to evaluate a specific configureation we can use these "static stresses"
        #the result is we can get very close to the actual analytical solution-- but we cannot generalize
        #the fewer DOF the system has, the better it will perform, but the less we can generalize
        # stress1=np.ones_like(stress1)
        # stress2=np.ones_like(stress1)*0
        # stress3=np.ones_like(stress1)


        pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
        pt_y = Variable(torch.from_numpy(y).float(), requires_grad=True).to(device)
        pt_z = Variable(torch.from_numpy(z).float(), requires_grad=True).to(device)
        stress1 = Variable(torch.from_numpy(stress1).float(), requires_grad=True).to(device)
        stress2 = Variable(torch.from_numpy(stress2).float(), requires_grad=True).to(device)
        stress3 = Variable(torch.from_numpy(stress3).float(), requires_grad=True).to(device)
        time = Variable(torch.from_numpy(time).float(), requires_grad=True).to(device)
        x = torch.reshape(pt_x, (num_collocation, 1))
        y = torch.reshape(pt_y, (num_collocation, 1))
        z = torch.reshape(pt_z, (num_collocation, 1))
        stress1 = torch.reshape(stress1, (num_collocation, 1))
        stress2 = torch.reshape(stress2, (num_collocation, 1))
        stress3 = torch.reshape(stress3, (num_collocation, 1))
        time = torch.reshape(time, (num_collocation, 1))

        if optimizer_version==2:
            stress_divergence_loss, linear_elasticity_loss, volume_loss = f(time,stress1,stress2,stress3, x, y, z, net, epoch, num_collocation)  # output of f(x,t)
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

    time_to_evaluate=1

    direction1=0
    direction2=.5
    direction3=0

    a=True
    single_eval=True
    trainBC=a
    trainPDE=a
    make_plots=True

    plot_surf=False

    random_runs=False
    setup = True
    evaluate = True

    quick_eval=False

    if time_to_evaluate>1:
        time_to_evaluate=1.00

    with open('triaxial_pinn.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["direction1", "direction2","direction3", "np.max(disp_x + x)",
                         "np.max(disp_y + y)", "np.max(disp_z + z)"])
        while time_to_evaluate<=1.001:



            if setup:

                net = PDE_Net()
                net = net.to(device)
                optimizer = torch.optim.Rprop(net.parameters(),lr=.0002,step_sizes=[1e-8,1])
                #optimizer = torch.optim.Adam(net.parameters())
                optimizer2 = torch.optim.LBFGS(net.parameters(),lr=.05, history_size=150, max_iter=15,line_search_fn='strong_wolfe')


                E = torch.ones(1, device=device) * 1
                E.to(device)


                if trainPDE:
                    start = time.perf_counter()
                    losses = train_PDE(net, optimizer, optimizer_version=2)
                    end = time.perf_counter()
                    torch.save(net.state_dict(), "PDE_triaxial.pt")
                    plt.plot(losses)
                    # plt.yscale('log')
                    #losses = train_PDE(BC_net, net, optimizer2, optimizer_version=1)
                    torch.save(net.state_dict(), "PDE_triaxial.pt")
                    plt.plot(losses)
                    plt.yscale('log')
                    print(f"Trained model in {end - start:0.4f} seconds")
                    plt.show()

                net.load_state_dict(torch.load("PDE_triaxial.pt"))

                if quick_eval:
                    a = 5
                    s1 = np.linspace(0, 1, a)
                    s2 = np.linspace(0, 1, a)
                    s3 = np.linspace(0, 1, a)
                    x = np.array([1])
                    y = np.array([1])
                    z = np.array([1])

                    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
                    pt_y = Variable(torch.from_numpy(y).float(), requires_grad=True).to(device)
                    pt_z = Variable(torch.from_numpy(z).float(), requires_grad=True).to(device)
                    pt_s1 = Variable(torch.from_numpy(s1).float(), requires_grad=True).to(device)
                    pt_s2 = Variable(torch.from_numpy(s2).float(), requires_grad=True).to(device)
                    pt_s3 = Variable(torch.from_numpy(s3).float(), requires_grad=True).to(device)
                    pt_x_collocation = torch.reshape(pt_x, (np.size(x), 1))
                    pt_y_collocation = torch.reshape(pt_y, (np.size(x), 1))
                    pt_z_collocation = torch.reshape(pt_z, (np.size(x), 1))
                    s1_pt = torch.reshape(pt_s1, (a, 1))
                    s2_pt = torch.reshape(pt_s2, (a, 1))
                    s3_pt = torch.reshape(pt_s3, (a, 1))

                    for i in s1_pt:
                        for j in s2_pt:
                            for k in s3_pt:
                                s1 = torch.reshape(i, (1, 1))
                                s2 = torch.reshape(j, (1, 1))
                                s3 = torch.reshape(k, (1, 1))
                                distance_function = dist(s1, s2, s3, pt_x_collocation, pt_y_collocation,pt_z_collocation)
                                outputs_BC = get_BC(s1, s2, s3, pt_x_collocation, pt_y_collocation, pt_z_collocation)
                                outputs_PDE = distance_function * net(s1, s2, s3, pt_x_collocation, pt_y_collocation, pt_z_collocation)
                                outputs = outputs_PDE + outputs_BC

                                outputs = outputs.data.cpu().detach().numpy()
                                disp_x = outputs[:, 0]
                                disp_y = outputs[:, 1]
                                disp_z = outputs[:, 2]

                                writer.writerow([s1.data.cpu().detach().numpy()[0,0], s2.data.cpu().detach().numpy()[0,0],s3.data.cpu().detach().numpy()[0,0], disp_x[0]+1,disp_y[0]+1,disp_z[0]+1])
                                #print("wrote row",i,j,k)
                    quit()



            if evaluate:
                x = np.random.uniform(0, 1, 398)
                x = np.append(x, [.5, .5])
                y = np.random.uniform(-1, .95, 398)
                y = np.append(y, [0, 1])
                z = np.random.uniform(0, 1, 398)
                z = np.append(z, [.5, .5])

                if plot_surf:
                    # x = np.array([0,0,0,0,1,1,1,1])
                    # y = np.array([0,0,1,1,0,0,1,1])
                    # z = np.array([0,1,0,1,0,1,0,1])

                    #or we can do this:
                    x = np.array([1])
                    y = np.array([1])
                    z = np.array([1])

                pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
                pt_y = Variable(torch.from_numpy(y).float(), requires_grad=True).to(device)
                pt_z = Variable(torch.from_numpy(z).float(), requires_grad=True).to(device)
                pt_x_collocation = torch.reshape(pt_x, (np.size(x), 1))
                pt_y_collocation = torch.reshape(pt_y, (np.size(x), 1))
                pt_z_collocation = torch.reshape(pt_z, (np.size(x), 1))



                #t = torch.ones_like(pt_x_collocation) * direction1 * scale*time_to_evaluate
                t=torch.ones_like(pt_x_collocation)*time_to_evaluate
                s1=torch.ones_like(pt_x_collocation)*time_to_evaluate*direction1
                s2 = torch.ones_like(pt_x_collocation) * time_to_evaluate * direction2
                s3 = torch.ones_like(pt_x_collocation) * time_to_evaluate * direction3

                distance_function =dist(s1,s2,s3,pt_x_collocation,pt_y_collocation,pt_z_collocation)

                outputs_BC = get_BC(s1,s2,s3,pt_x_collocation,pt_y_collocation,pt_z_collocation)

                outputs_PDE = distance_function * net(s1,s2,s3,pt_x_collocation, pt_y_collocation, pt_z_collocation)
                outputs = outputs_PDE + outputs_BC

                outputs = outputs.data.cpu().detach().numpy()
                disp_x = outputs[:, 0]
                disp_y = outputs[:, 1]
                disp_z = outputs[:, 2]


            lam=np.max(disp_x + x)/np.max(x)
            vol=np.max(disp_x + x) * np.max(disp_y + y) * np.max(disp_z + z)
            nh_stress=1/3*(lam*lam-1/(lam))
            # print("current volume is: ",vol," Original is 2")
            # print("NH Stress needed for this deformation ",nh_stress," Actual stress is ",time_to_evaluate)
            #
            print("time ", time_to_evaluate)
            print("max_x", np.max(disp_x + x))
            print("max_y", np.max(disp_y + y))
            print("max_z", np.max(disp_z + z))

            print("max_vol",np.max(disp_x + x)* np.max(disp_y + y+1)*np.max(disp_z + z))

            writer.writerow([time_to_evaluate, direction2, np.max(disp_x + x),
                             np.max(disp_y + y), np.max(disp_z + z)])


            time_to_evaluate+=.1

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

            #plt.show()
            #time_to_evaluate += 1 / 100

            if single_eval == True:
                #plt.show()
                stresses = (direction1, direction2, direction3)
                l1, l2, l3 = fsolve(analytic_equations, (1, 1, 1), args=stresses)

                print("analytical solution", l1, l2*2-1, l3)
                plt.show()
                quit()
            plt.close()
            setup=False

        stresses = (direction1, direction2, direction3)
        l1, l2, l3 = fsolve(analytic_equations, (1, 1, 1), args=stresses)

        print("analytical solution", l1, l2*2-1, l3)



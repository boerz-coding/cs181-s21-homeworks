import numpy as np

data = [(0., 0., 0.),
        (0., 0.5, 0.),
        (0., 1., 0.),
        (0.5, 0., 0.5),
        (0.5, 0.5, 0.5),
        (0.5, 1., 0.5),
        (1., 0., 1.),
        (1., 0.5, 1.),
        (1., 1., 1.)]

alpha = 20

W1 = alpha * np.array([[1., 0.], [0., 1.]])
W2 = alpha * np.array([[0.1, 0.], [0., 1.]])
W3 = alpha * np.array([[1., 0.], [0., 0.1]])


def compute_loss(W):
    ## TO DO
    loss = 0
    N=np.shape(data)[0]
    for i in range(N):
        dlossup=0
        dlossdown=0
        for j in range(N):
            if(j !=i ):
                dlossup+=np.exp(-((data[j][0]-data[i][0])**2*W[0][0]+2*(data[j][0]-data[i][0])*(data[j][1]-data[i][1])*W[0][1]+(data[j][1]-data[i][1])**2*W[1][1]))*(data[j][2]-data[i][2])
                dlossdown+=np.exp(-((data[j][0]-data[i][0])**2*W[0][0]+2*(data[j][0]-data[i][0])*(data[j][1]-data[i][1])*W[0][1]+(data[j][1]-data[i][1])**2*W[1][1]))
        loss+=(dlossup/dlossdown)**2
    return loss


print(compute_loss(W1))
print(compute_loss(W2))
print(compute_loss(W3))

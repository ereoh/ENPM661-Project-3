import matplotlib.pyplot as plt
import numpy as np

X0 = np.array((0))
Y0= np.array((0))
U1 = np.array((2))
V1 = np.array((-2))

fig, ax = plt.subplots()
q0 = plt.quiver(X0, Y0, U1, V1,units='xy' ,scale=1,color= 'r',headwidth = 1,headlength=0)

# Action 1 (U1,V1) on Node0     --> Node1

Node1=[X0+U1, Y0+V1]
X1=X0+U1
Y1=Y0+V1

print('Node1: ')

print(Node1)



#Action 2 (U2,V2) on Node1     --> Node2

U2= np.array((3))
V2 = np.array((-2))

q1 = plt.quiver(X1, Y1, U2, V2,units='xy' ,scale=1)

Node2=[X1+U2, Y1+V2]

print('Node2: ')

print(Node2)

#Action 3 (U3,V3) on Node1    --> Node3

U3 = np.array((3))
V3 = np.array((-1))

q2 = ax.quiver(X1, Y1, U3, V3,units='xy' ,scale=1)

Node3=[X1+U3, Y1+V3]

print('Node3: ')
print(Node3)




#Action 4  (U4,V4) on Node1   --> Node4

U4 = np.array((1))
V4 = np.array((-3.5))

q3 = ax.quiver(X1, Y1, U4, V4,units='xy' ,scale=1)

Node4=[X1+U4, Y1+V4]

print('Node4: ')

print(Node4)




plt.grid()

ax.set_aspect('equal')

plt.xlim(-10,10)
plt.ylim(-10,10)

plt.title('How to plot a vector in matplotlib ?',fontsize=10)

plt.savefig('how_to_plot_a_vector_in_matplotlib_fig3.png', bbox_inches='tight')

plt.show()
plt.close()
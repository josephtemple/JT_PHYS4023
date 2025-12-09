"""
step2.py

Same as step1, except now we handle the Brownian rotation in a sensible manner (i.e. as a random walk
in theta, rather than resetting the angle each step.)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
plt.rcParams['figure.dpi'] = 150

# potential and its gradient
def V(x, y, A = 10):
    return A * (x**4 - x**2 + y**4 - y**2)

def del_V(x, y, A = 10):
    dVds = lambda s: A *(4 * s**3 - 2 * s)
    dVdx = dVds(s=x)
    dVdy = dVds(s=y)

    return np.array([dVdx, dVdy])

# defining the box
L = 1
n_pts = 500
x_range = np.linspace(-L, L, n_pts)
y_range = np.linspace(-L, L, n_pts)

X, Y = np.meshgrid(x_range, y_range)

# Particle class
class Particles:
    def __init__(self, x_i, y_i, v_i, theta_i):
        self.n = len(x_i)
        self.x = x_i
        self.y = y_i

        self.v = v_i
        self.theta = theta_i

        self.radius = 0.02
        self.diffusion_rate = 5
        self.mobility = .05
        self.dt = 1/500

    def update(self):
        # brownian-ly change direction
        self.theta += np.sqrt(2 * self.diffusion_rate * self.dt) * np.random.normal(size = self.n)

        # collision detection, checking each pair of particles
        for i in range(self.n):
            for j in range(i+1, self.n):
                delta_x = self.x[j] - self.x[i]
                delta_y = self.y[j] - self.y[i]

                center_distance = np.sqrt(delta_x**2 + delta_y**2)
                
                if center_distance < 2*self.radius:
                    # flip particle velocities
                    self.theta[i] = -self.theta[i]
                    self.theta[j] = -self.theta[j]

                    # move particles apart 
                    nx = delta_x / center_distance 
                    ny = delta_y / center_distance

                    dist_to_move = 2*self.radius - center_distance
                    dx = dist_to_move * nx
                    dy = dist_to_move * ny

                    self.x[i] = self.x[i] - dx / 2
                    self.y[i] = self.y[i] - dy / 2

                    self.x[j] = self.x[j] + dx / 2
                    self.y[j] = self.y[j] + dy / 2

            # boundary conditions: reverse along that direction
            if self.x[i] - self.radius < -L:
                self.x[i] = -L + self.radius
                self.theta[i] = np.pi - self.theta[i] # (vx, vy) -> (-vx, vy) 
            if self.x[i] + self.radius > L:
                self.x[i] = L - self.radius
                self.theta[i] = np.pi - self.theta[i] 
            if self.y[i] - self.radius < -L:
                self.y[i] = -L + self.radius
                self.theta[i] = -self.theta[i] # (vx, vy) -> (vx, -vy) 
            if self.y[i] + self.radius > L:
                self.y[i] = L - self.radius
                self.theta[i] = -self.theta[i] 

        self.theta %= 2*np.pi

    def step(self):
        gradV = del_V(self.x, self.y)
        Fx, Fy = -self.mobility * gradV

        self.x += (self.v * np.cos(self.theta) + Fx) * self.dt
        self.y += (self.v * np.sin(self.theta) + Fy) * self.dt

    def get_position(self):
        return self.x, self.y


# defining particle ensemble 
N = 100
particle_spawn_lim = 0.7 * L
xs, ys = np.random.uniform(-particle_spawn_lim, particle_spawn_lim, (2,N))
thetas = np.random.uniform(0, 2*np.pi, N)
v0 = 2
vs = np.ones(N) * v0

particle_ensemble = Particles(xs, ys, vs, thetas)

# plot potential with a colormap
fig = plt.figure()
ax = plt.axes(xlim=(-L, L), ylim=(-L, L))
fig.suptitle(r"Active Brownian motion of an ensemble, low $D$ low $\mu$")
potential = V(X, Y, A = 1)
im = ax.imshow(potential, extent=(-L, L, -L, L), cmap = 'PiYG')
ax.set_xticks([])
ax.set_yticks([])
cbar = fig.colorbar(im, ax=ax, label="Potential")

# create particles as circles on that potential
particles = []
for i in range(N):
        particle = Circle((xs[i], ys[i]), particle_ensemble.radius, facecolor='white', edgecolor='black')
        ax.add_patch(particle)
        particles.append(particle)

# animation functions and actually animating
def init():
    """initialize animation"""
    return particles

def animate(t):
    """update particle velocities and take steps in new direction"""
    particle_ensemble.update()
    particle_ensemble.step()
    pos = particle_ensemble.get_position()
    for i, c in enumerate(particles):
        c.center = (pos[0][i], pos[1][i])
    return particles

anim = animation.FuncAnimation(fig, animate, init_func = init, frames=300, interval=5, blit=True)

save = True
if save:
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(script_dir)
    writer = animation.FFMpegWriter(fps=30, codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-profile:v', 'baseline', '-level', '3.0'])
    anim.save(f'{script_dir}/videos/step2lowDhlowmu.mp4', writer=writer)

plt.show()
plt.close()
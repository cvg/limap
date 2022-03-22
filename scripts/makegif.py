import os, sys
import imageio

def main():
    fname_output = "output.gif"

    folder = 'tmp/vis_refinement'
    img_list = []
    n_states = 18
    for i in range(n_states):
        fname = os.path.join(folder, 'state_{0}.png'.format(i))
        img = imageio.imread(fname)
        img_list.append(img)

    imageio.mimsave(fname_output, img_list, duration=0.25)

if __name__ == "__main__":
    main()



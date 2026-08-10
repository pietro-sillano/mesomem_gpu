import os,sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

DESCRIPTION = """
This script initializes a squared hexagonal grid configuration for LAMMPS simulations.
Available arguments:

--path  path of the output files
--N     Number of particles for square edge
--d     Distance between particles
--PBC   if insert PBC system is generated
--plot  to plot the lattice
--h     print this help message

Example of script with arguments

python planar_lattice.py --path . --N 50 --PBC --d 0.80 --plot
"""

def parse_arguments():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(usage=DESCRIPTION,description=DESCRIPTION)

    parser.add_argument("--path", type=str, help="Path to save the configuration files",required=True)

    parser.add_argument("--N", type=int, help="Number of particles for square edge",default=50)

    parser.add_argument("--d", type=float, help="Distance between particles",default=0.863)

    parser.add_argument("--PBC", action="store_true", help="configuration for PBC system",default=True)

    parser.add_argument("--plot", action="store_true", help="Generate a plot of the configuration",default=False)

    return parser.parse_args()

def write_positions(outfile,x,y,z,mux,muy,muz,Lbox_x,Lbox_y,Lbox_z,type):

    # Define LAMMPS atom data file content

    # atom-ID atom-type x y z diameter density q mux muy muz molecule-ID

    # density = 1.0 / (4.0/3.0*np.pi)
    density = 1.0
    diameter = 1.0
    mol_id = 1
    q = 1.0 # charge


    lammps_data = f"""

    {len(x)} atoms

    1 atom types

    {-Lbox_x/2.0:.6f} {Lbox_x/2.0:.6f} xlo xhi
    {-Lbox_y/2.0:.6f} {Lbox_y/2.0:.6f} ylo yhi
    {-Lbox_z/2.0:.6f} {Lbox_z/2.0:.6f} zlo zhi


    Atoms \n
    """
    # atom-ID atom-type x y z diameter density q mux muy muz molecule-ID

    # Append atom positions to the LAMMPS data
    idx = 1
    for i in range(x.shape[0]):
        lammps_data += f"{idx:d} {type[i]:d} {x[i]:.6f} {y[i]:.6f} {z[i]:.6f} {diameter} {density} {q} {mux[i]:.6f} {muy[i]:.6f} {muz[i]:.6f} {int(mol_id):d} \n"
        idx = idx + 1

    # Write the LAMMPS data to a file
    with open(outfile, "w") as file:
        file.write(lammps_data)

    print(f"LAMMPS data file {outfile} has been generated.")


def create_dir(path):
    base_name = os.path.basename(path)
    parent_dir = os.path.dirname(path)

    # If the directory exists, rename it with a GROMACS-style suffix
    if os.path.exists(path):
        suffix_num = 0
        new_name = f"#{base_name}#"

        # Check for existing folders with the pattern and increment as needed
        while os.path.exists(os.path.join(parent_dir, new_name)):
            suffix_num += 1
            new_name = f"#{base_name}.{suffix_num}#"

        renamed_path = os.path.join(parent_dir, new_name)
        os.rename(path, renamed_path)
        print(f"Existing folder renamed to: {renamed_path}")

    # Create the new directory
    os.makedirs(path)
    print(f"New directory created: {path}")

    return path

def plot_hexagonal_grid(hex_centers):
    """
    Plot the hexagonal grid using matplotlib.
    """
    plt.figure()
    plt.plot(hex_centers[:, 0], hex_centers[:, 1], 'o')
    plt.title("Hexagonal Grid of Particles")
    plt.axis('equal')
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid()
    plt.show()

def generate_hex_grid(nx, ny, min_diam):
    """
    Generates a hexagonal grid of points.

    Args:
        nx (int): Number of hexagons in the x direction.
        ny (int): Number of hexagons in the y direction.
        min_diam (float): Distance between particle centers.

    Returns:
        tuple: A tuple containing:
            - hex_centers (np.ndarray): An array of shape (N, 2) where N is the total
              number of hexagons, containing the (x, y) coordinates of each hexagon center.
            - dummy (None): A placeholder to match the original function's return signature.
    """

    x_coords = []
    y_coords = []

    for row in range(ny):
        y_offset = row * min_diam * np.sqrt(3) / 2.0
        # Determine x-offset for staggering
        if row % 2 == 1: # Odd rows are shifted
            x_start_offset = min_diam / 2.0
        else: # Even rows are not shifted
            x_start_offset = 0.0

        for col in range(nx):
            x_coords.append(x_start_offset + col * min_diam)
            y_coords.append(y_offset)

    hex_centers = np.array([x_coords, y_coords]).T

    return hex_centers, None

def test_hex_grid_generation():
    """
    Tests the generate_hex_grid function for correctness.
    """
    print("\n--- Running Tests for generate_hex_grid ---")

    # Test Case 1: Small grid (2x2)
    nx, ny = 2, 2
    min_diam = 1.0
    hex_centers, _ = generate_hex_grid(nx, ny, min_diam)

    expected_num_points = nx * ny
    if len(hex_centers) == expected_num_points:
        print(f"Test Case 1 ({nx}x{ny}): Number of points correct ({len(hex_centers)}).")
    else:
        print(f"Test Case 1 ({nx}x{ny}): FAILED. Expected {expected_num_points} points, got {len(hex_centers)}.")
        return False

    # Check distances between adjacent points
    # Expected points for 2x2 with min_diam=1.0:
    # (0.0, 0.0), (1.0, 0.0)
    # (0.5, 0.866), (1.5, 0.866)
    
    # Check horizontal distance in first row
    dist1 = np.linalg.norm(hex_centers[1] - hex_centers[0])
    if np.isclose(dist1, min_diam):
        print(f"Test Case 1 ({nx}x{ny}): Horizontal distance in row 1 correct ({dist1:.3f}).")
    else:
        print(f"Test Case 1 ({nx}x{ny}): FAILED. Incorrect horizontal distance in row 1: {dist1:.3f}.")
        return False

    # Check distance between staggered points (e.g., (0,0) and (0.5, 0.866))
    # Point at (0.0, 0.0) is hex_centers[0]
    # Point at (0.5, 0.866) is hex_centers[2]
    dist2 = np.linalg.norm(hex_centers[2] - hex_centers[0])
    expected_staggered_dist = min_diam
    if np.isclose(dist2, expected_staggered_dist):
        print(f"Test Case 1 ({nx}x{ny}): Staggered distance correct ({dist2:.3f}).")
    else:
        print(f"Test Case 1 ({nx}x{ny}): FAILED. Incorrect staggered distance: {dist2:.3f}. Expected {expected_staggered_dist:.3f}")
        return False

    # Test Case 2: Larger grid (5x3) with different min_diam
    nx, ny = 5, 3
    min_diam = 0.5
    hex_centers, _ = generate_hex_grid(nx, ny, min_diam)

    expected_num_points = nx * ny
    if len(hex_centers) == expected_num_points:
        print(f"Test Case 2 ({nx}x{ny}): Number of points correct ({len(hex_centers)}).")
    else:
        print(f"Test Case 2 ({nx}x{ny}): FAILED. Expected {expected_num_points} points, got {len(hex_centers)}.")
        return False

    # Check distances for a few points in the larger grid
    # Example: Check horizontal distance in a non-first row
    # The points in the second row (index 5, 6 for 5x3 grid) should be 0.5 apart horizontally
    dist3 = np.linalg.norm(hex_centers[6] - hex_centers[5])
    if np.isclose(dist3, min_diam):
        print(f"Test Case 2 ({nx}x{ny}): Horizontal distance in row 2 correct ({dist3:.3f}).")
    else:
        print(f"Test Case 2 ({nx}x{ny}): FAILED. Incorrect horizontal distance in row 2: {dist3:.3f}.")
        return False

    print("--- All generate_hex_grid tests passed! ---")
    return True


def main():
    """
    Main function to generate and save hexagonal grid configuration.
    """
    # Run tests first
    if not test_hex_grid_generation():
        print("Tests failed. Exiting.")
        sys.exit(1) # Exit if tests fail

    args = parse_arguments()


    INTERPARTICLE_DISTANCE = args.d

    N_ATOMS = args.N * args.N
    diameter = INTERPARTICLE_DISTANCE

    # Generate hexagonal grid
    hex_centers, _ = generate_hex_grid(nx=args.N, ny=args.N, min_diam=diameter)
    
    
    # Resize the lattice to square one
    # lenght is 2, height is 1.73 for every hexagon, every 4 exagon you have to remove one column.    
    # resize = args.N // 4 
    LATTICE_SQUARE = True
    if LATTICE_SQUARE:
        ymax = np.max(hex_centers[:, 1])
        mask = np.where(hex_centers[:, 0] < ymax)
        hex_centers = hex_centers[mask]

    # Define box dimensions
    if args.PBC:
        edge_border = args.d
        Lbox_x = (np.max(hex_centers[:, 0]) -
                np.min(hex_centers[:, 0])) + edge_border
        Lbox_y = (np.max(hex_centers[:, 1]) -
                np.min(hex_centers[:, 1])) + edge_border
        Lbox_z = Lbox_x
        
        print(Lbox_x)

    else:
        # add some padding to the edges, for free standing membrane
        factor = 1.5
        Lbox_x = (np.max(hex_centers[:, 0]) - np.min(hex_centers[:, 0])) * factor
        Lbox_y = (np.max(hex_centers[:, 1]) - np.min(hex_centers[:, 1])) * factor
        Lbox_z = Lbox_x

    # shift the lattice center at the origin
    x = hex_centers[:, 0] - Lbox_x / 2.0
    y = hex_centers[:, 1] - Lbox_y / 2.0
    z = np.zeros_like(hex_centers[:, 0])
    # rng = np.random.default_rng()
    # phase = rng.integers(low=1, high=3, size=hex_centers.shape[0])
    phase   = np.ones(hex_centers.shape[0],dtype=np.int8)


    # particles orientation set to positive in z direction
    mux = np.ones_like(x) * 0.0
    muy = np.ones_like(x) * 0.0
    muz = np.ones_like(x) * 1.0

    outfile = os.path.join(args.path, f'lattice_d_{INTERPARTICLE_DISTANCE:.2f}_N_{N_ATOMS}')

    # Save particle positions
    write_positions(outfile, x, y, z, mux, muy, muz, Lbox_x, Lbox_y, Lbox_z,phase)


    print(f"Configuration saved to: {outfile}")
    print(f"Run LAMMPS simulation with following command")
    print(f"mpirun -np 4 lmp_membrane20Feb24 -in in.planar_membrane -v N {N_ATOMS}")

    # Plot configuration if requested
    if args.plot:
        plot_hexagonal_grid(hex_centers)

if __name__ == "__main__":
    main()

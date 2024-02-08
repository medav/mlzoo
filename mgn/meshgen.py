
import torch
from . import graphnet as GNN


def gen_2d_tri_mesh(nx, ny):
    xs = torch.linspace(0, 1, nx)
    ys = torch.linspace(0, 1, ny)
    xv, yv = torch.meshgrid(xs, ys, indexing='xy')
    mesh_pos = torch.stack([xv.flatten(), yv.flatten()], dim=1)

    num_cells = 2 * (nx - 1) * (ny - 1)
    cells = torch.zeros((num_cells, 3), dtype=torch.int64)
    idx = 0

    for r in range(ny - 1):
        for c in range(nx - 1):
            cells[idx, 0] = r * nx + c
            cells[idx, 1] = r * nx + c + 1
            cells[idx, 2] = (r + 1) * nx + c
            idx += 1

            cells[idx, 0] = r * nx + c + 1
            cells[idx, 1] = (r + 1) * nx + c + 1
            cells[idx, 2] = (r + 1) * nx + c
            idx += 1

    return cells, mesh_pos

def gen_3d_tet_mesh(nx, ny, nz):
    xs = torch.linspace(0, 1, nx)
    ys = torch.linspace(0, 1, ny)
    zs = torch.linspace(0, 1, nz)

    xv, yv, zv = torch.meshgrid(xs, ys, zs, indexing='xy')
    mesh_pos = torch.stack([xv.flatten(), yv.flatten(), zv.flatten()], dim=1)

    num_tets = 6 * (nx - 1) * (ny - 1) * (nz - 1)
    cells = torch.zeros((num_tets, 4), dtype=torch.int64)
    idx = 0

    for x in range(nx - 1):
        for y in range(ny - 1):
            for z in range(nz - 1):
                base_index = x * ny * nz + y * nz + z
                nodes = {
                    '000': base_index,
                    '001': base_index + 1,
                    '010': base_index + nz,
                    '011': base_index + nz + 1,
                    '100': base_index + ny * nz,
                    '101': base_index + ny * nz + 1,
                    '110': base_index + ny * nz + nz,
                    '111': base_index + ny * nz + nz + 1
                }

                tets = [
                    [nodes['000'], nodes['001'], nodes['010'], nodes['100']],
                    [nodes['001'], nodes['010'], nodes['100'], nodes['101']],
                    [nodes['010'], nodes['100'], nodes['101'], nodes['110']],
                    [nodes['001'], nodes['010'], nodes['101'], nodes['110']],
                    [nodes['001'], nodes['101'], nodes['110'], nodes['111']],
                    [nodes['001'], nodes['011'], nodes['101'], nodes['111']]
                ]

                for tet in tets:
                    cells[idx] = torch.tensor(tet, dtype=torch.int64)
                    idx += 1

    return cells, mesh_pos

if __name__ == '__main__':
    cells, mesh_pos = gen_2d_tri_mesh(3, 5)

    print(cells)
    print(mesh_pos)

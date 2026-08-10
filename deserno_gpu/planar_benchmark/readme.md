# various kokkos combination

--- Best configurations per N ---
  N=   9600  TPS=1539.1  neigh=full newton=off comm=device sort=device OMP=1
  N=  38400  TPS=706.4  neigh=half newton=on comm=device sort=host OMP=4
  N= 153600  TPS=283.4  neigh=half newton=on comm=device sort=host OMP=4



The important thing is comm:device, sort:device, neigh half


moving everything to device is quite important, that is why in mesomem is slower due to the hybrid pair style. I should open an issue.


neigh full newton on seems to work for smaller systems

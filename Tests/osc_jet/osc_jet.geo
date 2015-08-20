Point(1) = {0, -0.25, 0, 1.0};
Extrude {1, 0, 0} {
  Point{1};
}
Extrude {0, 0.5, 0} {
  Line{1};
}
Physical Line("Inlet") = {3};
Physical Line("Bottom") = {1};
Physical Line("Outlet") = {4};
Physical Line("Top") = {2};
Physical Surface("Fluid") = {5};
Transfinite Surface {5};
Transfinite Line {1, 2} = 20 Using Progression 1;
Transfinite Line {3, 4} = 20 Using Bump 1.9;
Recombine Surface {5};

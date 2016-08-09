.highlightlang:: none

**************
Catalyst guide
**************

PyFR/Catalyst
=============

This version of PyFR includes a plugin that is intended to be used with a
custom ParaView Catalyst in situ visualization library.

Configuration
-------------

To use the plugin, a new section of PyFR's .ini file must be created:
``[soln-plugin-catalyst]``.

Ini file
^^^^^^^^

The ``soln-plugin-catalyst`` accepts a number of parameters.

- *nsteps*: the number of simulation timesteps to progress before performing
  visualization.  Lower numbers produce a more fluid, responsive visualization
  experience, but increase the overhead as well.
- *outputfile*: unused?
- *hostname*: the hostname that the plugin will try to connect to for the
  `pvserver` process.
- *port*: port to connect to on the aforementioned *hostname*.
- *metadata_out*: Boolean.  output per-timestep metadata as the simulation
  runs.  Note that this adds global communciation to every visualization
  invocation.
- *isovalues*: python list containing the isovalues to use for computing
  isosurfaces.
- *eye*: 3-element python list specifying where the camera falls (X, Y, Z)
- *ref*: 3-element python list specifying where the camera focuses (X, Y, Z)
- *vup*: 3-element python list specifying the view orientation for the camera.
  Generally we think of a human standing staight up as ``[ 0, 1, 0 ]``; a human
  laying on the ground would be ``[ 0, 0, 1 ]`` (or ``-1``, depending on which
  direction they are laying).

``eye`` / ``ref`` / ``vup`` can be omitted to have ParaView automatically
choose the camera positions.

Environment variables
^^^^^^^^^^^^^^^^^^^^^

The Catalyst plugin reads a number of environment variables to allow easier
configuration.  Environment variables *override* the values specified in the
.ini file.

- *PV_HOSTNAME*: the hostname with running `pvserver` to connect to.
- *PV_PORT*: port number for `pvserver` connection.

These are recommended in supercomputing environments, as the host and
port names are typically not available until the job launches.

Software setup
--------------

The Catalyst configuration requires a unique software stack.

- `ParaView 4.4, +modifications 
  <https://gitlab.kitware.com/tjcorona/paraview.git>`_ and `sc2015demo` branch.
- a recent X server
- `Pre-release VTK-m <https://gitlab.kitware.com/tjcorona/vtk-m.git>`_ and the
  `sc2015demo` branch.
- The C++ Catalyst/PyFR plugin in this repository.

X server
^^^^^^^^

You will need to run an X server on the server nodes to be able to
generate images.

Known issues
------------

- ``eye``/``ref``/``vup`` are not yet implemented!

Sample Workflow
---------------

1. Get everything setup and working with a scaled-down version of your desired
   run.  Set the ``metadata_out`` parameter to ``True`` in your .ini file.

2. Examine your job's output for ``[catalyst]`` lines.  The ``range`` lines
   give the per-timestep scalar range of the data given to the isosurfacing
   algorithm.  The ``world bounds`` lines give per-timestep information on
   where the domain lies in world space.

3. Revisit the ``isovalues`` parameter based on the ``range`` metadata.
   You will need at least one isovalue within the range for a visible
   isosurface to appear.

4. Consider setting the ``eye``/``ref``/``vup`` parameters based on the ``world
   bounds`` metadata.

5. Repeat from step 1 until satisfied.

6. Set ``metadata_out`` to ``False`` in your .ini file.

7. Run using your large-scale input.

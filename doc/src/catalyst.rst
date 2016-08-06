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
- *isovalues*: python list containing the isovalues to use for computing
  isosurfaces.
- *eye*: 3-element python list specifying where the camera falls (X, Y, Z)
- *ref*: 3-element python list specifying where the camera focuses (X, Y, Z)
- *vup*: 3-element python list specifying the view orientation for the camera.
  Generally we think of a human standing staight up as ``[ 0, 1, 0 ]``; a human
  laying on the ground would be ``[ 0, 0, 1 ]`` (or ``-1``, depending on which
  direction they are laying).

``eye`` / ``ref`` / ``vup`` can be omitted to have ParaView automatically
choose the camera positions (but see 'Known Issues', below).

Environment variables
^^^^^^^^^^^^^^^^^^^^^

The Catalyst plugin reads a number of environment variables to allow easier
configuration.  Environment variables *override* the values specified in the
.ini file.

- *PV_HOSTNAME*: the hostname with running `pvserver` to connect to.
- *PV_PORT*: port number for `pvserver` connection.

Using these is recommended in supercomputing environments, as the host and
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

- The bounds of the domain are not reported to ParaView correctly.  This
  causes ParaView to choose poor camera positions.  Use the camera
  configuration .ini parameters as a workaround.

Sample Workflow
---------------

1. Get everything setup and working with a 2 GPU test case.  Particularly on a
   supercomputer, this will require many iterations debugging environment
   variable settings (e.g. ``LD_LIBRARY_PATH``.)

2. Examine standard output when running the job for lines such as:
   ``[catalyst] world bounds0: [ -4.48, 53.4, -32.5, 32.5, -32.4, 32.4, ]``
   This line comes from the C++ code ("``[catalyst]``").  It reports the
   world-space bounds of the isosurface[s] on process 0.

3. Revisit the ``eye``/``ref``/``vup`` parameters given the above world space
   bounds.  ``[10.0, 0.0, 0.0]``

4. Examine standard output when running the job for lines such as::

  [catalyst] range1: 0.994779--1.00526

5. Revisit the ``isovalues`` parameters given the above bounds on the scalar.

6. Rerun.

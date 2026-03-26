PyNutil
=======

PyNutil is a Python library for brain-wide quantification and spatial analysis
of features in serial section images from the brain. It aims to replicate the
Quantifier feature of the Nutil software (RRID: SCR_017183).

PyNutil integrates outputs from atlas registration software and image
segmentation workflows to produce atlas-based quantifications, 3D point clouds,
and 3D heatmaps of brain-derived data.

.. image:: ../assets/PyNutil_fig1.png
   :alt: Overview figure for the PyNutil workflow
   :width: 100%

.. warning::

   PyNutil is still under development and the API is subject to change.

The documentation below brings together installation notes, the basic workflow,
practical examples, and the generated API reference.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api_index

Highlights
----------

* Use BrainGlobe atlases or custom atlas volumes in ``.nrrd`` format.
* Read registration data from QuickNII, VisuAlign, DeepSlice, or
  BrainGlobe registration outputs.
* Quantify binary segmentations or intensity images against atlas regions.
* Export point clouds for MeshView and interpolated NIfTI volumes for
  siibra explorer or ITK-SNAP.

For more background on the QUINT workflow, see
`QUINT workflow documentation <https://quint-workflow.readthedocs.io/en/latest/>`_.

Index & Search
--------------

* :ref:`genindex`
* :ref:`search`

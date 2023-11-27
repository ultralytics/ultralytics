fastreid.modeling
===========================

.. automodule:: fastreid.modeling
    :members:
    :undoc-members:
    :show-inheritance:

Model Registries
-----------------

These are different registries provided in modeling.
Each registry provide you the ability to replace it with your customized component,
without having to modify fastreid's code.

Note that it is impossible to allow users to customize any line of code directly.
Even just to add one line at some place,
you'll likely need to find out the smallest registry which contains that line,
and register your component to that registry.


.. autodata:: fastreid.modeling.BACKBONE_REGISTRY
.. autodata:: fastreid.modeling.META_ARCH_REGISTRY
.. autodata:: fastreid.modeling.REID_HEADS_REGISTRY

"""
SCPMel: A Pytorch-friendly package providing tools for building 1D Physically
Informed Neural Networks (PINN) models with a variety of explicit time integration scheme.
============================================================================================
SCPMel is composed of the following submodules:
-----------
Using any of these submodule requires an explicit import. For instance,
``import SCPMel.integrate''.
::
 integrate                    --- Integration routines through Pytorch
 les                          --- Filters and spectral routines
 reduction                    --- Dimensionality reduction routines
 sampling                     --- Sampling routines
 training                     --- Pytorch extra utility routines
 utils                        --- Additionals tools and plot utilities 

"""

__all__ = ["integrate", "les", "reduction", "sampling", "training", "utils"]
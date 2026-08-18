# -*- coding: utf-8 -*-
"""MagAnomalyPicker Tools QGIS plugin."""

def classFactory(iface):  # pylint: disable=invalid-name
    from .mag_picker_plugin import MagPickerPlugin
    return MagPickerPlugin(iface)

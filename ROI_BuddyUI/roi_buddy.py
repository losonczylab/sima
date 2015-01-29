#! python
import os
import sys
from sys import path
path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from os.path import join, dirname, isdir

import numpy as np
from datetime import datetime
from scipy.spatial import ConvexHull
from scipy.cluster.hierarchy import average, fcluster
from scipy.stats import mode
from shapely.geometry import MultiPolygon, Polygon
from skimage import transform as tf
import itertools as it
from random import shuffle
import warnings as wa

from roiBuddyUI import Ui_ROI_Buddy
from importROIsWidget import Ui_importROIsWidget

import sima
from sima.imaging import ImagingDataset
from sima.ROI import ROIList, ROI, mask2poly, poly2mask
from sima.segment import ca1pc
from sima.misc import TransformError, estimate_array_transform, \
    estimate_coordinate_transform

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from guidata import qthelpers
from guiqwt.plot import ImageDialog
from guiqwt.tools import FreeFormTool, InteractiveTool, \
    RectangleTool, RectangularShapeTool, SelectTool
from guiqwt.builder import make
from guiqwt.shapes import PolygonShape, EllipseShape
from guiqwt.events import setup_standard_tool_filter, PanHandler
from guiqwt.image import ImageItem


icon_filepath = \
    './icons/'

if isdir('/data/'):
    data_path = '/data/'
else:
    data_path = ''


def debug_trace():
    '''Set a tracepoint in the Python debugger that works with Qt'''

    from PyQt4.QtCore import pyqtRemoveInputHook
    from pudb import set_trace
    pyqtRemoveInputHook()
    set_trace()


def jaccard_index(roi1, roi2):
    """Calculates the Jaccard index of two rois.
    Defined as the ratio of the size of the intersection to the size of the
    union of the two ROIs in pixels.

    Parameters
    ----------
    roi1, roi2 : shapely.geometry.Polygon

    """
    union = 0
    intersection = 0

    roi1_polys = [p for p in roi1]
    roi2_polys = [p for p in roi2]

    while(len(roi1_polys)):
        for first_poly in roi1_polys:
            z = np.array(first_poly.exterior.coords)[0, 2]

            co_planar_polys1 = [first_poly]
            for other_poly in [p for p in roi1_polys if p is not first_poly]:
                if np.array(other_poly.exterior.coords)[0, 2] == z:
                    co_planar_polys1.append(other_poly)
            p1 = MultiPolygon(co_planar_polys1)

            co_planar_polys2 = []
            for p in roi2_polys:
                if np.array(p.exterior.coords)[0, 2] == z:
                    co_planar_polys2.append(p)
            p2 = MultiPolygon(co_planar_polys2)

            union += p1.union(p2).area
            intersection += p1.intersection(p2).area

            for p in co_planar_polys1[::-1]:
                roi1_polys.remove(p)
            for p in co_planar_polys2[::-1]:
                roi2_polys.remove(p)

    while(len(roi2_polys)):
        for extra_polygon in roi2_polys:
            z = np.array(extra_polygon.exterior.coords)[0, 2]

            co_planar_polys = [extra_polygon]
            for other_poly in [
                    p for p in roi2_polys if p is not extra_polygon]:
                if np.array(other_poly.exterior.coords)[0, 2] == z:
                    co_planar_polys.append(other_poly)
            p0 = MultiPolygon(co_planar_polys)

            union += p0.area

            for p in co_planar_polys[::-1]:
                roi2_polys.remove(p)

    jaccard = intersection / union
    return jaccard


class PanTool(InteractiveTool):
    """Allows panning with the left mouse button.
    http://spykeutils.readthedocs.org/en/0.4.0/

    """
    TITLE = "Pan"
    ICON = os.path.abspath(os.path.join(icon_filepath, "transform-move.png"))
    CURSOR = Qt.OpenHandCursor

    def setup_filter(self, baseplot):
        filter = baseplot.filter
        start_state = filter.new_state()
        PanHandler(filter, Qt.LeftButton, start_state=start_state)

        return setup_standard_tool_filter(filter, start_state)


class EllipseTool(RectangularShapeTool):
    # TODO: Modify this such that it draws like an ImageJ ellipse?

    TITLE = "Ellipse"
    ICON = "ellipse_shape.png"

    def create_shape(self):
        shape = EllipseShape(0, 0, 1, 1)
        self.set_shape_style(shape)
        return shape, 0, 1

    def handle_final_shape(self, shape):
        shape.switch_to_ellipse()
        super(EllipseTool, self).handle_final_shape(shape)


class RoiBuddy(QMainWindow, Ui_ROI_Buddy):
    """Instance of the ROI Buddy Qt interface."""
    def __init__(self):
        """
        Initialize the application
        """

        # initialize the UI and parent class
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.setWindowTitle('ROI Buddy')

        # define actions for menu bar and ROI manager
        self.define_actions()

        self.create_menu()
        # self.create_status_bar()

        self.viewer = self.initialize_display_window()

        # filter for Esc button behavior
        self.viewer.keyPressEvent = self.viewer_keyPressEvent
        self.viewer.wheelEvent = self.viewer_wheelEvent
        self.viewer.add_tool(PanTool)

        self.initialize_roi_manager()

        self.initialize_contrast_panel()

        self.connectSignals()

        # initialize base image
        self.base_im = None

        # initialize the mode
        self.mode = 'edit'

        self.colors_dict = {}

        # deactivate buttons until a t-series is added
        self.toggle_button_state(False)

        self.toggle_button_state(True)

        self.disable_drawing_tools()

    def viewer_keyPressEvent(self, event):
        """Esc button filter -- prevent application from crashing"""
        if (event.key() == Qt.Key_Escape):
            event.ignore()
        else:
            ImageDialog.keyPressEvent(self.viewer, event)

    def viewer_wheelEvent(self, event):
        """Capture scroll events to toggle the base image z-plane"""
        active_tSeries = self.tSeries_list.currentItem()
        delta = event.delta()
        if delta < 0:
            if active_tSeries.active_plane + 1 >= active_tSeries.num_planes:
                return
            self.plane_index_box.setValue(active_tSeries.active_plane + 1)
        else:
            if active_tSeries.active_plane - 1 < 0:
                return
            self.plane_index_box.setValue(active_tSeries.active_plane - 1)

    def create_menu(self):
        self.file_menu = self.menuBar().addMenu("&File")

        qthelpers.add_actions(self.file_menu, [self.add_tseries_action,
                                               self.auto_add_tseries_action,
                                               None,
                                               self.save_rois_action,
                                               self.save_all_action,
                                               None,
                                               self.edit_label_action,
                                               self.add_tags_action,
                                               self.clear_tags_action,
                                               self.edit_tags_action,
                                               None,
                                               self.merge_rois,
                                               self.unmerge_rois,
                                               None,
                                               self.quit_action])

    def create_status_bar(self):
        self.statusBar().addWidget(QLabel(""), 1)

    def define_actions(self):
        # File menu actions
        self.quit_action = qthelpers.create_action(
            self, "&Quit", triggered=self.close, shortcut="Ctrl+Q",
            tip="Close ROI Buddy")

        self.add_tseries_action = qthelpers.create_action(
            self, "&Add t-series", triggered=self.add_tseries,
            shortcut="Ctrl+O", tip="Add a single t-series .sima folder")

        self.auto_add_tseries_action = qthelpers.create_action(
            self, "&Auto add t-series", triggered=self.add_tseries_by_tag,
            tip="Auto add .sima folders with a tag string filter")

        self.save_rois_action = qthelpers.create_action(
            self, "&Save current ROIs",
            triggered=lambda: self.save([self.tSeries_list.currentItem()]),
            shortcut="Ctrl+S",
            icon=QIcon(QString(icon_filepath + "document-save-5.png")),
            tip="Save the current ROI set to file.")

        self.save_all_action = qthelpers.create_action(
            self, "&Save all ROIs",
            triggered=lambda: self.save(
                [self.tSeries_list.item(i) for i in range(
                    self.tSeries_list.count())]),
            icon=QIcon(QString(icon_filepath + "document-save-all.png")
                       ), tip="Save all ROI sets with the same label")

        # #ROI Manager actions
        self.delete_action = qthelpers.create_action(
            self, "&Delete ROI", triggered=self.delete, shortcut="Del",
            icon=QIcon(QString(icon_filepath + "edit-delete-2.png")),
            tip="Remove the selected ROIs")
        self.delete_action.setShortcuts(["D", "Del", "Backspace"])

        self.edit_label_action = qthelpers.create_action(
            self, "&Edit Label", triggered=self.edit_label,
            icon=QIcon(QString(icon_filepath + "document-sign.png")),
            tip="Edit the ROI 'label' attribute")

        self.add_tags_action = qthelpers.create_action(
            self, "&Add Tags", triggered=self.add_tags, shortcut="T",
            icon=QIcon(QString(icon_filepath + "flag-green.png")),
            tip="Add tags to the selected ROIs")

        self.clear_tags_action = qthelpers.create_action(
            self, "&Remove Tags", triggered=self.clear_tags, shortcut="C",
            icon=QIcon(QString(icon_filepath + "flag-red.png")),
            tip="Remove tags from the selected ROIs")

        self.edit_tags_action = qthelpers.create_action(
            self, "&Edit Tags", triggered=self.edit_tags,
            icon=QIcon(QString(icon_filepath + "edit.png")),
            tip="Edit tags for the selected ROI")

        self.merge_rois = qthelpers.create_action(
            self, "&Merge", triggered=self.merge_ROIs, shortcut="M",
            icon=QIcon(QString(icon_filepath + "insert-link.png")),
            tip="Merge ROIs")

        self.unmerge_rois = qthelpers.create_action(
            self, "&Unmerge", triggered=self.unmerge_ROIs, shortcut="U",
            icon=QIcon(QString(icon_filepath + "edit-cut.png")),
            tip="Unmerge an ROI")

        self.edit_roi_action = qthelpers.create_action(
            self, "&Edit", triggered=self.edit_roi, shortcut="E",
            tip="Edit the selected ROI")
        self.addAction(self.edit_roi_action)

        self.randomize_colors_action = qthelpers.create_action(
            self, "&Randomize", triggered=self.randomize_colors, shortcut="R",
            icon=QIcon(QString(icon_filepath + "colorize.png")),
            tip="Randomize ROI colors")

        self.activate_freeform_tool_action = qthelpers.create_action(
            self, "&Activate Freeform Tool",
            triggered=self.activate_freeform_tool, shortcut="F",
            tip="Activate the Freeform Tool")
        self.addAction(self.activate_freeform_tool_action)

        self.activate_selection_tool_action = qthelpers.create_action(
            self, "&Activate Selection Tool",
            triggered=self.activate_selection_tool, shortcut="S",
            tip="Activate Selection tool")
        self.addAction(self.activate_selection_tool_action)

        self.debug_action = qthelpers.create_action(
            self, "&Debug", triggered=debug_trace, shortcut="F10")
        self.addAction(self.debug_action)

    def connectSignals(self):

        # toggle the edit or align mdoe
        self.modeSelection.buttonClicked.connect(self.toggle_mode)

        # buttons for adding and removing t-series from list
        self.add_tseries_button.clicked.connect(self.add_tseries)
        self.remove_tseries_button.clicked.connect(self.remove_tseries)

        # selection in tSeries_list changes
        self.tSeries_list.currentItemChanged.connect(
            self.toggle_active_tSeries)

        # toggle the active roi set
        self.active_rois_combobox.activated.connect(self.toggle_rois)

        # add/delete roi sets
        self.new_set_button.clicked.connect(self.new_roi_set)
        self.delete_set_button.clicked.connect(self.delete_roi_set)

        # z-plane selection
        self.plane_index_box.valueChanged.connect(self.toggle_plane)

        # Channel selection
        self.baseImage_list.activated.connect(self.toggle_base_image)
        self.processed_checkbox.clicked.connect(self.toggle_base_image)

        # show/hide ROIs
        self.show_ROIs_checkbox.stateChanged.connect(self.toggle_show_rois)
        self.show_all_checkbox.stateChanged.connect(self.toggle_show_all)

        # save ROIs buttons
        self.save_current_rois_button.clicked.connect(
            lambda: self.save([self.tSeries_list.currentItem()]))

        # Run registration
        self.register_rois_button.clicked.connect(self.register_rois)

        # Propagate tags
        self.propagate_tags_button.clicked.connect(self.propagate_tags)

        # Import transformed ROIs
        self.import_rois_button.clicked.connect(self.import_rois)

    def closeEvent(self, event):

        reply = QMessageBox.question(
            self, 'Message', "Do you want to save all ROIs?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Yes)

        if reply == QMessageBox.Yes:
            self.save([self.tSeries_list.item(i) for i in
                       range(self.tSeries_list.count())])
            event.accept()
        elif reply == QMessageBox.No:
            event.accept()
        else:
            event.ignore()

    def initialize_display_window(self):
        """
        Configure the main display window with a guiqwt ImageDialog object
        """

        # initialize the viewer
        viewer = ImageDialog(edit=False, toolbar=True,
                             wintitle='Experiment Image Display')
        # Add the freeform tool
        viewer.add_tool(FreeFormTool)
        viewer.add_tool(EllipseTool)
        viewer.add_tool(RectangleTool)
        # Remove the grid from the item list manager
        viewer.get_plot().get_items()[0].set_private(True)
        # add viewer to the display frame layout
        self.displayFrame.layout().addWidget(viewer)

        # remove useless buttons
        for i in range(3, 18)[::-1]:
            viewer.toolbar.removeAction(viewer.toolbar.actions()[i])

        self.plot = viewer.get_plot()
        for tool in viewer.tools:
            if type(tool) == FreeFormTool:
                self.freeform_tool = tool
            elif type(tool) == SelectTool:
                self.selection_tool = tool
            elif type(tool) == RectangleTool:
                self.rectangle_tool = tool
            elif type(tool) == EllipseTool:
                self.ellipse_tool = tool

        return viewer

    def enable_drawing_tools(self):
        self.freeform_tool.deactivate()
        self.freeform_tool.action.setEnabled(True)
        self.rectangle_tool.deactivate()
        self.rectangle_tool.action.setEnabled(True)
        self.ellipse_tool.deactivate()
        self.ellipse_tool.action.setEnabled(True)

    def disable_drawing_tools(self):
        self.freeform_tool.deactivate()
        self.freeform_tool.action.setEnabled(False)
        self.rectangle_tool.deactivate()
        self.rectangle_tool.action.setEnabled(False)
        self.ellipse_tool.deactivate()
        self.ellipse_tool.action.setEnabled(False)

    def initialize_roi_manager(self):

        # add the ROI manager to the roiListFame
        itemListPanel = self.viewer.get_itemlist_panel()
        layout = QGridLayout()

        layout.addWidget(itemListPanel)
        self.itemListFrame.setLayout(layout)
        itemListPanel.show()

        # This is the ROI manager toolbar
        toolbar = itemListPanel.findChild(QToolBar)

        # remove the default delete actions -- doesn't work with freeform tool
        toolbar.removeAction(toolbar.actions()[-1])

        # adding actions to the ROI manager toolbar
        toolbar.addAction(self.delete_action)
        toolbar.addAction(self.edit_label_action)
        toolbar.addAction(self.add_tags_action)
        toolbar.addAction(self.clear_tags_action)
        toolbar.addAction(self.edit_tags_action)
        toolbar.addAction(self.merge_rois)
        toolbar.addAction(self.unmerge_rois)
        toolbar.addAction(self.randomize_colors_action)

    def initialize_contrast_panel(self):
        """
        Configure the contrast panel and set its expansion policy
        """
        layout = QGridLayout()
        contrastPanel = self.viewer.get_contrast_panel()
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(
            contrastPanel.sizePolicy().hasHeightForWidth())
        contrastPanel.setSizePolicy(sizePolicy)
        layout.addWidget(contrastPanel, 0, 0)

        self.lookupTableFrame.setLayout(layout)
        contrastPanel.show()

    def toggle_button_state(self, enabled=True):
        # for when there are no t-series in the list
        self.remove_tseries_button.setEnabled(enabled)

        self.active_rois_combobox.clear()
        self.baseImage_list.clear()

        [button.setEnabled(enabled) for button in
         self.save_rois_widget.children() if
         isinstance(button, QPushButton)]

        [action.setEnabled(enabled) for action in
         [self.save_rois_action, self.save_all_action]]

        [button.setEnabled(enabled) for button in
         self.show_rois_widget.children() if
         isinstance(button, QCheckBox)]

        [button.setEnabled(enabled) for button in
         self.mode_selection_widget.children() if
         isinstance(button, QRadioButton) or
         isinstance(button, QPushButton)]

        [button.setEnabled(enabled) for button in
         self.roi_set_widget.children() if
         isinstance(button, QComboBox) or
         isinstance(button, QPushButton)]

        [button.setEnabled(enabled) for button in
         self.channelSelectionFrame.children() if
         isinstance(button, QComboBox) or
         isinstance(button, QCheckBox) or
         isinstance(button, QSpinBox)]

        [action.setEnabled(enabled) for action in
         self.viewer.get_itemlist_panel().findChild(QToolBar).actions()]

        if self.mode == 'edit':
            self.show_all_checkbox.setEnabled(False)
            self.register_rois_button.setEnabled(False)
            self.propagate_tags_button.setEnabled(False)
            self.import_rois_button.setEnabled(enabled)

    def remove_rois(self, roi_list):
        self.freeform_tool.shape = None
        for item in reversed(self.plot.items):
            if item in roi_list:
                self.plot.del_item(item)

    def show_rois(self, tSeries, show_in_list=None):
        if self.mode == 'edit':
            tSeries.transform_rois()
        for roi in tSeries.roi_list:
            if roi.coords[0][0, 2] == tSeries.active_plane:
                roi.show(show_in_list)

    def hide_rois(self, show_in_list=None):
        for item in self.plot.items:
            if isinstance(item, UI_ROI):
                item.hide(show_in_list)

    def delete(self):
        # Custom delete slot is necessary because of bug when interacting with
        # freeform tool, otherwise if you start a polygon, don't close it,
        # delete it, then use the freeform tool again, it fails.
        self.freeform_tool.shape = None
        items = self.plot.get_selected_items()
        # Don't delete the base image
        items = [item for item in items if not isinstance(item, ImageItem)]
        self.plot.unselect_all()
        if self.mode == 'align':
            for item in items:
                item.parent.roi_list.remove(item)
        self.plot.del_items(items)
        self.plot.replot()

    def activate_freeform_tool(self):
        if self.mode == 'edit':
            self.freeform_tool.activate()
            self.freeform_tool.shape = None

    def activate_selection_tool(self):
        self.selection_tool.activate()

    def toggle_mode(self, button):

        active_tSeries = self.tSeries_list.currentItem()

        if button is self.align_mode_radiobutton and self.mode is 'edit':

            active_tSeries.update_rois()

            y_lims = [x + self.base_im.data.shape[0]
                      for x in self.plot.get_axis_limits(0)]
            x_lims = [x + self.base_im.data.shape[1]
                      for x in self.plot.get_axis_limits(2)]

            self.selection_tool.activate()
            self.edit_label_action.setEnabled(False)
            self.add_tags_action.setEnabled(False)
            self.clear_tags_action.setEnabled(False)
            self.edit_tags_action.setEnabled(False)
            self.disable_drawing_tools()
            self.active_rois_combobox.setEnabled(False)
            self.show_all_checkbox.setEnabled(True)
            self.register_rois_button.setEnabled(True)
            self.propagate_tags_button.setEnabled(True)
            self.import_rois_button.setEnabled(False)

            self.mode = 'align'
            self.toggle_roi_editing_state(False)
            active_tSeries.transform_rois(active_tSeries)

            active_tSeries.show()

            # Preserve the zoom and aspect ratio
            self.plot.set_axis_limits(0, y_lims[0], y_lims[1])
            self.plot.set_axis_limits(2, x_lims[0], x_lims[1])

        if button is self.edit_mode_radiobutton and self.mode is 'align':

            y_lims = self.plot.get_axis_limits(0)
            x_lims = self.plot.get_axis_limits(2)

            self.enable_drawing_tools()
            self.edit_label_action.setEnabled(True)
            self.add_tags_action.setEnabled(True)
            self.clear_tags_action.setEnabled(True)
            self.edit_tags_action.setEnabled(True)
            self.active_rois_combobox.setEnabled(True)
            self.show_all_checkbox.setCheckState(False)
            self.show_all_checkbox.setEnabled(False)
            self.register_rois_button.setEnabled(False)
            self.propagate_tags_button.setEnabled(False)
            self.import_rois_button.setEnabled(True)

            self.mode = 'edit'
            self.toggle_roi_editing_state(True)
            active_tSeries.transform_rois()

            active_tSeries.show()

            y_lims = [x - self.base_im.data.shape[0] for x in y_lims]
            x_lims = [x - self.base_im.data.shape[1] for x in x_lims]

            # Preserve the zoom and aspect ratio
            self.plot.set_axis_limits(0, y_lims[0], y_lims[1])
            self.plot.set_axis_limits(2, x_lims[0], x_lims[1])

        self.hide_rois(show_in_list=False)
        self.toggle_show_rois()

    def add_tseries(self):
        """Select the .sima file for the imaging dataset to load
        Adds the directory to the tSeriesList widget

        """

        sima_path = str(QFileDialog.getExistingDirectory(
            None, 'Select the .sima folder', data_path,
            QFileDialog.ShowDirsOnly))

        if sima_path is not '':
            if not self.tSeries_list.count():
                self.toggle_button_state(True)
            try:
                tSeries = UI_tSeries(sima_path, parent=self)
            except IOError:
                print('Invalid path, skipping: ' + sima_path)
                if not self.tSeries_list.count():
                    self.toggle_button_state(False)
            else:
                self.tSeries_list.setCurrentItem(tSeries)

    def add_tseries_by_tag(self):
        """Select a root folder, then recursively search all subdirectories for
        the .sima folders containing tag

        """

        root_path = str(QFileDialog.getExistingDirectory(
            None, 'Select the parent directory in which to look for ' +
            '.sima folders', data_path,
            QFileDialog.ShowDirsOnly))

        if root_path is not '':
            tag, ok = QInputDialog.getText(
                self, 'Auto-add t-series by tag',
                'Enter the tag.  All .sima folders recursively found in the ' +
                'parent directory that contain this tag will be added.')

            if ok:
                paths = []
                for root, dirnames, _ in os.walk(root_path):
                    for directory in dirnames:
                        if directory.endswith('.sima') \
                                and str(tag) in directory:
                            paths.append(join(root, directory))

                paths.sort()
                for p in paths:
                    UI_tSeries(p, parent=self)

                if self.tSeries_list.count():
                    self.toggle_button_state(True)
                    self.tSeries_list.setCurrentRow(0)

    def remove_tseries(self):
        """Remove the currently selected tSeries from the list"""
        active_tSeries = self.tSeries_list.currentItem()
        self.tSeries_list.takeItem(self.tSeries_list.row(active_tSeries))
        rois_to_remove = [
            item for item in self.plot.items if
            (isinstance(item, UI_ROI) and item.parent == active_tSeries) or
            (isinstance(item, PolygonShape) and not isinstance(item, UI_ROI))]
        self.remove_rois(rois_to_remove)
        if not self.tSeries_list.count():
            # TODO: is this check necessary?
            try:
                self.plot.del_item(self.base_im)
            except AttributeError:
                # base_im has already been deleted
                pass
            self.toggle_button_state(False)
            self.plot.replot()

    def initialize_roi_set_list(self, tSeries):

        self.active_rois_combobox.clear()
        self.active_rois_combobox.insertItems(0, tSeries.roi_sets)
        if tSeries.active_rois is not None:
            current_idx = tSeries.roi_sets.index(tSeries.active_rois)
            self.active_rois_combobox.setCurrentIndex(current_idx)

    def initialize_base_image_list(self, tSeries):

        self.baseImage_list.clear()
        self.baseImage_list.insertItems(0, tSeries.dataset.channel_names)
        current_idx = tSeries.dataset.channel_names.index(
            tSeries.active_channel)
        self.baseImage_list.setCurrentIndex(current_idx)

    def initialize_z_planes_box(self, tSeries):
        self.plane_index_box.setWrapping(True)
        self.plane_index_box.setRange(0, tSeries.num_planes - 1)
        self.plane_index_box.setValue(tSeries.active_plane)

    def new_roi_set(self):
        """Add a new ROI Set to the imaging dataset"""

        active_tSeries = self.tSeries_list.currentItem()
        active_tSeries.update_rois()
        if active_tSeries.active_rois is not None:
            save = QMessageBox.question(
                self, 'Save Changes', 'Save changes to the current rois?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

            if save == QMessageBox.Yes:
                self.save([self.tSeries_list.currentItem()])

        roi_set_name, ok = QInputDialog.getText(
            self, 'ROI Set Name',
            'Enter the name of the new ROI set:')

        if ok:
            self.remove_rois(active_tSeries.roi_list)
            active_tSeries.roi_sets.append(str(roi_set_name))
            active_tSeries.active_rois = str(roi_set_name)
            self.initialize_roi_set_list(active_tSeries)
            active_tSeries.initialize_rois()
            self.hide_rois(show_in_list=False)
            self.enable_drawing_tools()
            self.plot.replot()

    def delete_roi_set(self):
        """Delete an ROI set from the rois.pkl file"""

        active_tSeries = self.tSeries_list.currentItem()

        delete = QMessageBox.question(
            self, 'Delete ROIs',
            "Delete '{}' ROIs from the current tSeries?".format(
                active_tSeries.active_rois),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if delete == QMessageBox.Yes:
            rois_to_remove = [
                item for item in self.plot.items if
                (isinstance(item, UI_ROI) and item.parent == active_tSeries) or
                (isinstance(item, PolygonShape)
                    and not isinstance(item, UI_ROI))]
            self.remove_rois(rois_to_remove)
            active_tSeries.dataset.delete_ROIs(active_tSeries.active_rois)
            active_tSeries.roi_sets.remove(active_tSeries.active_rois)
            active_tSeries.active_rois = None
            self.initialize_roi_set_list(active_tSeries)

            if len(active_tSeries.roi_sets) == 0:
                self.disable_drawing_tools()
            else:
                self.toggle_rois()

            self.plot.replot()

    def toggle_base_image(self):
        name = str(self.baseImage_list.currentText())
        if name is not '':
            active_tSeries = self.tSeries_list.currentItem()
            y_lims = self.plot.get_axis_limits(0)
            x_lims = self.plot.get_axis_limits(2)

            active_tSeries.active_channel = name
            active_tSeries.show()

            self.plot.set_axis_limits(0, y_lims[0], y_lims[1])
            self.plot.set_axis_limits(2, x_lims[0], x_lims[1])
            self.plot.replot()

    def toggle_active_tSeries(self, current, previous):
        """Toggle the active t-series"""

        if current is None:
            return

        if self.mode is 'edit':
            if previous is not None:
                previous.update_rois()
            current.transform_rois()

        if self.mode is 'align':
            self.show_all_checkbox.setCheckState(False)
            current.transform_rois(current)

        self.hide_rois(show_in_list=False)
        self.initialize_base_image_list(current)
        self.initialize_roi_set_list(current)
        self.initialize_z_planes_box(current)

        current.show()

        self.show_rois(current, show_in_list=self.mode == 'edit')
        self.toggle_show_rois()

        self.plot.replot()

        if len(current.roi_sets) == 0:
            self.disable_drawing_tools()
        else:
            self.enable_drawing_tools()

    def toggle_rois(self):

        active_tSeries = self.tSeries_list.currentItem()
        active_tSeries.update_rois()
        active_rois = str(self.active_rois_combobox.currentText())
        if active_rois != active_tSeries.active_rois:
            if active_tSeries.active_rois is not None:

                save = QMessageBox.question(
                    self, 'Save Changes', 'Save changes to the current ROIs?',
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

                if save == QMessageBox.Yes:
                    self.save([self.tSeries_list.currentItem()])

            self.remove_rois(active_tSeries.roi_list)
            active_tSeries.active_rois = active_rois

            active_tSeries.initialize_rois()
            self.show_rois(active_tSeries, show_in_list=self.mode == 'edit')
            self.plot.replot()

    def toggle_plane(self):

        self.freeform_tool.shape = None
        active_tSeries = self.tSeries_list.currentItem()
        tSeries_list = [self.tSeries_list.item(i)
                        for i in range(self.tSeries_list.count())]
        if self.mode == 'edit':
            active_tSeries.update_rois()

        # Note: toggling planes changes the active plane for every UI_tSeries!
        for tSeries in tSeries_list:
            tSeries.active_plane = self.plane_index_box.value()
        active_tSeries.show()
        self.hide_rois()
        if self.show_ROIs_checkbox.checkState():
            if self.show_all_checkbox.checkState():
                self.show_all_checkbox.setChecked(False)
                self.show_all_checkbox.setChecked(True)
            else:
                active_tSeries.transform_rois(active_tSeries)
                self.show_rois(active_tSeries)
        self.plot.replot()

    def edit_label(self):
        """Edit the labels of the selected ROIs"""

        active_tSeries = self.tSeries_list.currentItem()

        rois = self.plot.get_selected_items()
        rois = active_tSeries.update_rois(rois)
        if not len(rois):
            self.plot.unselect_all()
            return

        label, ok = QInputDialog.getText(
            self, 'Edit Label', 'Enter the label to be associated with each ' +
            'ROI. Note that polygons with the same label are saved as a ' +
            'single ROI object')

        selected_labels = list(set([roi.label for roi in rois]))
        for roi in active_tSeries.roi_list:
            if roi.label is not None and roi.label in selected_labels and \
                    roi not in rois:
                rois.append(roi)

        for roi in rois:
            roi.label = label
            roi.update_name()

    def add_tags(self):
        """
        Add tags to the selected ROIs
        """
        active_tSeries = self.tSeries_list.currentItem()

        rois = self.plot.get_selected_items()
        rois = active_tSeries.update_rois(rois)
        if not len(rois):
            self.plot.unselect_all()
            return

        tags, ok = QInputDialog.getText(
            self, 'Add Tags', 'Enter the tag strings:')

        tags_list = [str(x).strip() for x in tags.split(",")]

        if ok:
            for roi in rois:
                # If two separate polygons have the same name (same ROI),
                # make sure you add tag to both!
                polys_to_tag = [r for r in active_tSeries.roi_list
                                if r.label == roi.label]
                for poly in polys_to_tag:
                    for tag in tags_list:
                        poly.tags.add(tag)
                    poly.update_name()

    def clear_tags(self):
        """
        Remove tags from the selected ROIs
        """

        active_tSeries = self.tSeries_list.currentItem()

        rois = self.plot.get_selected_items()
        rois = active_tSeries.update_rois(rois)
        if not len(rois):
            self.plot.unselect_all()
            return

        for roi in rois:
            # If two separate polygons have the same id (same ROI),
            # make sure to remove flags from both!
            polys_to_clear = [r for r in active_tSeries.roi_list
                              if r.label == roi.label]

            for poly in polys_to_clear:
                poly.tags = None
                poly.update_name()

    def edit_tags(self):
        """
        Manually edit the tags for one selected ROI
        Accepts one selection
        """

        active_tSeries = self.tSeries_list.currentItem()

        # TODO: this is arbitrary -- either give warning or just fail
        roi = self.plot.get_selected_items()[0]
        rois = active_tSeries.update_rois([roi])
        if not len(rois):
            self.plot.unselect_all()
            return
        roi = rois[0]
        # if isinstance(roi, PolygonShape) and not isinstance(roi, UI_ROI):
        #     roi = UI_ROI.convert_polygon(roi, active_tSeries)
        #     if roi is None:
        #         self.plot.unselect_all()
        #         return
        #     active_tSeries.update_rois()

        # TODO: is this check necessary?
        if roi is not None:

            tags_str = ''
            for tag in sorted(roi.tags):
                tags_str += str(tag) + ', '

            new_tags, ok = QInputDialog.getText(
                self, 'Edit tags', 'Enter the list of comma-separated tags:',
                QLineEdit.Normal, tags_str)
            split_tags = [str(x).strip() for x in new_tags.split(',')]
            # TODO: ADD VALIDATOR!

            polys_to_edit = [r for r in active_tSeries.roi_list
                             if r.label == roi.label]

            if ok:
                for poly in polys_to_edit:
                    poly.tags = split_tags
                    poly.update_name()

    def merge_ROIs(self):
        """
        In edit mode:
        Merge the selected ROIs and give them the same tags and label
        If they are contiguous, redraw the polygon to replace the original two
        If they are not contiguous, give them the same name.  They will be
        considered one ROI during the signal extraction

        In align mode:
        """

        # TODO: What if one of the selected polys is part of a multi-poly ROI?
        # Do you merge the tags to those other polys as well?

        if self.mode is 'edit':
            self.freeform_tool.shape = None
            active_tSeries = self.tSeries_list.currentItem()

            selected_rois = self.plot.get_selected_items()
            selected_rois = active_tSeries.update_rois(selected_rois)
            if not len(selected_rois):
                self.plot.unselect_all()
                return

            rois = []
            tags = []
            labels = []

            selected_labels = list(set([roi.label for roi in selected_rois]))
            for roi in active_tSeries.roi_list:
                if roi.label is not None and roi.label in selected_labels and \
                        roi not in selected_rois:
                    selected_rois.append(roi)

            for roi in selected_rois:
                # if isinstance(roi, PolygonShape) \
                #         and not isinstance(roi, UI_ROI):
                #     roi = UI_ROI.convert_polygon(roi, active_tSeries)
                #     if roi is None:
                #         self.plot.unselect_all()
                #         return
                #     active_tSeries.roi_list.append(roi)
                rois.append(roi.get_points().tolist())
                self.plot.del_item(roi)
                tags.extend(roi.tags)
                labels.append(roi.label)
            # eliminate redundant tags
            tags = list(set(tags))
            labels = list(set(labels))

            mask = poly2mask(rois, im_size=self.base_im.data.shape)
            polys = mask2poly(mask)

            if None in labels:
                labels.remove(None)

            if len(labels) == 1:
                label = labels[0]
            elif len(labels) > 1:
                label = ''
                for l in labels:
                    label += str(l) + '_'
                label = label.rstrip('_')
            else:
                label = active_tSeries.next_label()

            for poly in polys:
                hull = ConvexHull(np.array(poly.exterior.coords))
                hull_pts = hull.points[hull.vertices].tolist()
                new_roi = UI_ROI(parent=active_tSeries, points=hull_pts,
                                 tags=tags, id=None, label=label)
                self.plot.add_item(new_roi)

            active_tSeries.update_rois()

        if self.mode is 'align':
            # NOTE: if roi2 is part of a cluster, it is essentially removed
            # from that and added to the cluster of roi1 (it also gets the
            # tags from cluster 1)

            rois = self.plot.get_selected_items()
            if len(rois) != 2:
                QMessageBox.warning(self,
                                    'Invalid Merge',
                                    'Invalid selection, must select exactly ' +
                                    '2 ROIs for merge.',
                                    QMessageBox.Ok)
                return
            [roi1, roi2] = rois
            tSeries_list = [self.tSeries_list.item(i)
                            for i in range(self.tSeries_list.count())]

            idx1_count = 0
            if roi1.id is not None:
                for tSeries in tSeries_list:
                    for roi in tSeries.roi_list:
                        if roi.id == roi1.id:
                            idx1_count += 1
                            break
            idx2_count = 0
            if roi2.id is not None:
                for tSeries in tSeries_list:
                    for roi in tSeries.roi_list:
                        if roi.id == roi2.id:
                            idx2_count += 1
                            break

            if idx1_count > idx2_count:
                parent = roi1
                child = roi2
            else:
                parent = roi2
                child = roi1

            # in align mode can't merge two from the same t-series
            if child.parent == parent.parent:
                QMessageBox.warning(self,
                                    'Invalid Merge',
                                    'The parent and child are members ' +
                                    'of the same set.  Merge in edit mode',
                                    QMessageBox.Ok)
                return

            # if the parent's ID is already present in the child's set
            if parent.id is not None and parent.id in [r.id for r in
                                                       child.parent.roi_list]:
                QMessageBox.warning(self,
                                    'Invalid Merge',
                                    'The parent ID is already contained in ' +
                                    'the child ROI set. It may be necessary ' +
                                    'to merge ROIs in edit mode.',
                                    QMessageBox.Ok)
                return

            if parent.id is None:
                next_id = self.next_id()
                polys_to_update = [r for r in parent.parent.roi_list
                                   if r.label == parent.label]
                for poly in polys_to_update:
                    poly.id = next_id
                    poly.update_name()
                    poly.update_color()

            polys_to_update = [r for r in child.parent.roi_list
                               if r.label == child.label]
            for poly in polys_to_update:
                poly.id = parent.id
                poly.update_name()
                poly.update_color()

            self.plot.unselect_all()

        self.plot.replot()

    def unmerge_ROIs(self):

        if self.mode is 'edit':
            selected_roi = self.plot.get_selected_items()
            if len(selected_roi) != 1:
                return
            selected_roi = selected_roi[0]

            if isinstance(selected_roi, UI_ROI):
                active_tSeries = self.tSeries_list.currentItem()
                selected_roi.label = active_tSeries.next_label()
                selected_roi.id = None
                selected_roi.update_name()
                selected_roi.update_color()
            elif isinstance(selected_roi, PolygonShape):
                # PolygonShapes don't have a label or id and can't exist as
                # merged objects (without first being converted to a UI_ROI)
                pass

        elif self.mode is 'align':
            selected_roi = self.plot.get_selected_items()
            if len(selected_roi) > 1:
                return
            selected_roi = selected_roi[0]

            polys_to_update = [r for r in selected_roi.parent.roi_list
                               if r.label == selected_roi.label]
            for poly in polys_to_update:
                poly.id = None
                poly.update_name()
                poly.update_color()

        self.plot.replot()

    def randomize_colors(self):
        if self.mode == 'edit':
            self.tSeries_list.currentItem().update_rois()

        for color_idx in self.colors_dict:
            self.colors_dict[color_idx] = random_color()

        tSeries_list = [self.tSeries_list.item(i)
                        for i in range(self.tSeries_list.count())]

        n_rois = sum([len(tSeries.roi_list) for tSeries in tSeries_list])

        z_levels = range(1, n_rois + 1)
        shuffle(z_levels)

        idx = 0
        for tSeries in tSeries_list:
            for roi in tSeries.roi_list:
                roi.update_color()
                roi.setZ(z_levels[idx])
                idx += 1

        self.plot.unselect_all()
        self.plot.replot()

    def edit_roi(self):

        PADDING = 10

        if self.mode is 'edit':
            return

        selected_roi = self.plot.get_selected_items()
        if len(selected_roi) > 1:
            return
        selected_roi = selected_roi[0]

        self.edit_mode_radiobutton.click()
        self.tSeries_list.setCurrentItem(selected_roi.parent)

        (min_x, min_y, max_x, max_y) = selected_roi.polygons.bounds

        self.plot.set_axis_limits(0, max_y + PADDING, min_y - PADDING)
        self.plot.set_axis_limits(2, min_x - PADDING, max_x + PADDING)

        self.plot.replot()

    def toggle_roi_editing_state(self, value=True):
        """Lock or unlock the ROIs in each tSeries for editing."""
        tSeries_list = [self.tSeries_list.item(i)
                        for i in range(self.tSeries_list.count())]

        for tSeries in tSeries_list:
            for roi in tSeries.roi_list:
                roi.toggle_editing(value)

    def toggle_show_rois(self):
        state = self.show_ROIs_checkbox.checkState()
        active_tSeries = self.tSeries_list.currentItem()

        if self.base_im is not None:
            if state:
                if self.mode is 'align':
                    self.show_all_checkbox.setEnabled(True)
                    self.toggle_show_all()
                else:
                    active_tSeries.update_rois()
                self.show_rois(
                    active_tSeries, show_in_list=self.mode == 'edit')
            else:
                if self.mode is 'align':
                    self.show_all_checkbox.setEnabled(False)
                else:
                    active_tSeries.update_rois()
                self.hide_rois(show_in_list=None)

            self.plot.replot()

    def toggle_show_all(self):
        state = self.show_all_checkbox.checkState()
        active_tSeries = self.tSeries_list.currentItem()
        tSeries_list = list(set([self.tSeries_list.item(i) for i in
                                 range(self.tSeries_list.count())]).difference(
                            set([active_tSeries])))
        if state:
            for tSeries in tSeries_list:
                try:
                    tSeries.transform_rois(active_tSeries)
                except TransformError:
                    QMessageBox.warning(
                        self,
                        'Transform Error',
                        'Cannot show all, unable to register ROIs to ' +
                        'tSeries.\n' +
                        'source={}\n'.format(tSeries.dataset.savedir) +
                        'target={}'.format(active_tSeries.dataset.savedir),
                        QMessageBox.Ok)
                    self.show_all_checkbox.setChecked(False)
                    self.toggle_show_all()
                    return
                self.show_rois(tSeries, show_in_list=self.mode == 'edit')
        else:
            self.hide_rois(show_in_list=self.mode == 'edit')
            self.show_rois(active_tSeries, show_in_list=self.mode == 'edit')
        self.plot.replot()

    def save(self, tSeries_list):
        """Save the active ROIs for each t-series"""

        active_tSeries = self.tSeries_list.currentItem()

        if len(tSeries_list) > 1:
            msg = 'Enter the save label to apply to the active ROI sets in ' +\
                  'each t-series:'
        else:
            msg = 'Enter the save label to apply for the current ROIs:'

        roi_set_name, ok = QInputDialog.getText(
            self, 'Save name', msg, text=active_tSeries.active_rois)

        if ok:
            if self.mode == 'edit':
                active_tSeries.update_rois()
            for tSeries in tSeries_list:
                tSeries.save_rois(str(roi_set_name))

        self.initialize_roi_set_list(active_tSeries)

    def register_rois(self):

        def intersects(roi1, roi2):
            """
            returns boolean valued on whether roi1 intersects roi2
            """
            for p1 in roi1:
                for p2 in roi2:
                    if np.array(p1.exterior.coords)[0, 2] == \
                            np.array(p2.exterior.coords)[0, 2]:
                        if p1.intersects(p2):
                            return True
            return False

        if not self.show_all_checkbox.isChecked():
            self.show_all_checkbox.setChecked(True)

        if not self.show_all_checkbox.isChecked():
            return

        active_tSeries = self.tSeries_list.currentItem()
        tSeries_list = [self.tSeries_list.item(i) for i in
                        range(self.tSeries_list.count())]

        # launch roi_lock popup
        self.roi_lock_popup = lockROIsWidget(self, tSeries_list)
        if not self.roi_lock_popup.exec_():
            return

        # rois is the original UI_ROIs, roi_polygons are ROIs converted to
        # shapely polygons, and roi_names is roi names
        # these 3 dictionaries should be the same length and correspond to
        # the entries of clusters
        rois = {}  # keys are t-series
        roi_polygons = {}  # keys are t-series
        roi_names = {}  # keys are t-series
        for tSeries in tSeries_list:
            rois[tSeries] = []
            roi_polygons[tSeries] = []
            roi_names[tSeries] = []
            try:
                tSeries.transform_rois(active_tSeries)
            except TransformError:
                return
            for roi in tSeries.roi_list:
                name = roi.label
                if name is None or name not in roi_names[tSeries]:
                    rois[tSeries].append([roi])
                    points = np.round(roi.get_points(), 1)
                    z = np.empty((len(points), 1))
                    z.fill(roi.coords[0][0, 2])
                    roi_polygons[tSeries].append(
                        [Polygon(np.hstack((points, z)))])
                    roi_names[tSeries].append(name)
                else:
                    idx = roi_names[tSeries].index(name)
                    rois[tSeries][idx].append(roi)
                    points = np.round(roi.get_points(), 1)
                    z = np.empty((len(points), 1))
                    z.fill(roi.coords[0][0, 2])
                    roi_polygons[tSeries][idx].append(
                        Polygon(np.hstack((points, z))))

        # Cast each ROI as a MultiPolygon (one per name.  Note that each Multi-
        # Polygon might be comprised of Polygons with different z-coordinates)
        for tSeries in tSeries_list:
            for roi_idx, roi in enumerate(roi_polygons[tSeries]):
                multi_poly = MultiPolygon(roi)
                if not multi_poly.is_valid:
                    multi_poly = mask2poly(
                        poly2mask(multi_poly, active_tSeries.transform_shape))
                roi_polygons[tSeries][roi_idx] = multi_poly

        condensed_distance_matrix = []
        for setIdx, tSeries in enumerate(tSeries_list):
            for roi1_idx, roi1 in enumerate(roi_polygons[tSeries]):
                for roi2 in roi_polygons[tSeries][roi1_idx + 1:]:
                    condensed_distance_matrix.append(0.)
                for tSeries2 in tSeries_list[setIdx + 1:]:
                    for roi2 in roi_polygons[tSeries2]:
                        if intersects(roi1, roi2):
                            condensed_distance_matrix.append(
                                jaccard_index(roi1, roi2))
                        else:
                            condensed_distance_matrix.append(0)

        linkage = average(1. / (0.0000001 + np.array(
            condensed_distance_matrix)))
        clusters = fcluster(linkage, 4, criterion='distance')

        # Group ROIs by cluster
        idx = 0
        ROIs_by_cluster = {}
        for tSeries in tSeries_list:
            for roi in rois[tSeries]:
                if clusters[idx] in ROIs_by_cluster:
                    ROIs_by_cluster[clusters[idx]].extend(roi)
                else:
                    ROIs_by_cluster[clusters[idx]] = roi
                idx += 1

        # Check each cluster for locked ROIs and just propagate ids
        all_unlocked_rois = []
        used_ids = set([])
        for cluster_id, rois in ROIs_by_cluster.iteritems():
            locked_rois = [roi for roi in rois if roi.parent.roi_id_lock]
            if len(locked_rois):
                roi_id, _ = mode([roi.id for roi in locked_rois])
                roi_id = roi_id.tostring()
                for roi in rois:
                    if roi.parent.roi_id_lock:
                        continue
                    roi.id = roi_id
                    roi.update_name()
                    roi.update_color()
                try:
                    used_ids.add(int(roi_id))
                except ValueError:
                    continue
            else:
                all_unlocked_rois.append(rois)

        # For clusters with no locked ROIS, give them a unique id
        unique_id_gen = (num for num in it.count() if num not in used_ids)
        for next_id, rois in it.izip(unique_id_gen, all_unlocked_rois):
            for roi in rois:
                roi.id = str(next_id)
                roi.update_name()
                roi.update_color()

        self.randomize_colors()

    def propagate_tags(self):

        tags, ok = QInputDialog.getText(
            self, 'Tags To Propagate',
            'Enter the tag strings separated by commas:')

        tags_to_propagate = set([str(x).strip() for x in tags.split(",")])

        tags_dict = {}
        tSeries_list = [self.tSeries_list.item(i)
                        for i in range(self.tSeries_list.count())]
        for tSeries in tSeries_list:
            for roi in tSeries.roi_list:
                if roi.id is not None:
                    if roi.id in tags_dict:
                        tags_dict[roi.id] = tags_dict[roi.id].union(
                            roi.tags.intersection(tags_to_propagate))
                    else:
                        tags_dict[roi.id] = roi.tags.intersection(
                            tags_to_propagate)

        for tSeries in tSeries_list:
            for roi in tSeries.roi_list:
                if roi.id is not None:
                    roi.tags = roi.tags.union(tags_dict[roi.id])
                    roi.update_name()

    def import_rois(self):

        tSeries_list = [self.tSeries_list.item(i) for i in
                        range(self.tSeries_list.count())]
        if len(tSeries_list) < 2:
            QMessageBox.warning(self,
                                'Import Error',
                                'At least two imaging datasets must be ' +
                                'loaded in order to import ROIs.',
                                QMessageBox.Ok)
            return

        active_tSeries = self.tSeries_list.currentItem()
        active_tSeries.update_rois()
        if active_tSeries.active_rois is not None:
            save = QMessageBox.question(
                self, 'Save Changes', 'Save changes to the current rois?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

            if save == QMessageBox.Yes:
                self.save([self.tSeries_list.currentItem()])

        source_dataset, source_channel, target_channel, source_label, \
            target_label, copy_properties, auto_manual, reg_method, \
            poly_order, ok = \
            ImportROIsWidget.getParams(self)

        if not ok:
            return

        if auto_manual == 'Auto':
            anchor_label = None
        else:
            if '_REGISTRATION_ANCHORS' not in source_dataset.roi_sets or \
                    '_REGISTRATION_ANCHORS' not in active_tSeries.roi_sets:
                QMessageBox.warning(
                    self, 'Transform Error',
                    'Need to save _REGISTRATION_ANCHORS ROIList',
                    QMessageBox.Ok)
                return
            anchor_label = '_REGISTRATION_ANCHORS'

        try:
            if reg_method == 'polynomial':
                method_args = {'order': poly_order}
            else:
                method_args = {}

            active_tSeries.dataset.import_transformed_ROIs(
                source_dataset=source_dataset.dataset,
                method=reg_method,
                source_channel=source_channel,
                target_channel=target_channel,
                source_label=source_label,
                target_label=target_label,
                anchor_label=anchor_label,
                copy_properties=copy_properties,
                **method_args)
        except TransformError:
            QMessageBox.warning(self, 'Transform Error',
                                'Transformation failed', QMessageBox.Ok)
            return
        else:
            self.remove_rois(active_tSeries.roi_list)
            active_tSeries.roi_sets.append(target_label)
            active_tSeries.active_rois = target_label
            self.initialize_roi_set_list(active_tSeries)
            active_tSeries.initialize_rois()
            self.show_rois(active_tSeries, show_in_list=True)
            self.plot.replot()

    def next_id(self):
        """Return the next valid unused id across all tSeries"""

        tSeries_list = [self.tSeries_list.item(i)
                        for i in range(self.tSeries_list.count())]
        all_ids = [roi.id for tSeries in tSeries_list
                   for roi in tSeries.roi_list]
        return next_int(all_ids)


class UI_tSeries(QListWidgetItem):
    def __init__(self, sima_path, parent):
        # Try to load the dataset first, if it fails, don't add it to the panel
        self.dataset = ImagingDataset.load(sima_path)
        
        QListWidgetItem.__init__(
            self, QString(dirname(sima_path)), parent=parent.tSeries_list)

        self.parent = parent

        self.num_planes = self.dataset.frame_shape[0]
        self.shape = self.dataset.frame_shape[1:3]

        self.transform_shape = tuple([int(3 * x) for x in self.shape])
        self.active_channel = self.dataset.channel_names[0]
        self.active_plane = 0

        # Lock ROI ids
        self.roi_id_lock = False

        rois = self.dataset.ROIs

        self.roi_sets = rois.keys()
        try:
            self.active_rois = sima.misc.most_recent_key(rois)
        except ValueError:
            new_set = datetime.strftime(datetime.now(), '%Y-%m-%d-%Hh%Mm%Ss')
            self.active_rois = new_set
            self.roi_sets.append(new_set)

        # self.transforms is a dictionary of affine transformations.
        # Keys are other UI_tSeries objects
        # TODO: skimage.Transform objects
        self.transforms = {}

        self.initialize_base_images()
        self.initialize_rois()

    def initialize_base_images(self):
        self.base_images = {}
        for channel_name in self.dataset.channel_names:
            self.base_images[channel_name] = []
        for plane in self.dataset.time_averages:
            for ch_idx, ch_name in enumerate(self.dataset.channel_names):
                self.base_images[ch_name].append(
                    plane[:, :, ch_idx] / np.amax(plane[:, :, ch_idx]))

    def show(self):
        try:
            self.parent.plot.del_item(self.parent.base_im)
        except AttributeError:
            # base_im has already been deleted
            pass
        except ValueError:
            self.parent.base_im = None

        channel_name = self.active_channel
        plane_idx = self.active_plane

        if self.parent.processed_checkbox.isChecked():
            key = 'processed_' + channel_name
            if key not in self.base_images:
                channel_idx = self.dataset.channel_names.index(channel_name)
                self.base_images[key] = ca1pc._processed_image_ca1pc(
                    self.dataset, channel_idx=channel_idx, x_diameter=14,
                    y_diameter=7)
            data = self.base_images[key][plane_idx]
        else:
            data = self.base_images[channel_name][plane_idx]

        if self.parent.mode == 'align':
            trans = tf.AffineTransform(
                translation=tuple([-1 * x for x in self.shape[::-1]]))

            data = tf.warp(data, trans, output_shape=self.transform_shape,
                           mode='constant', cval=0)

        self.parent.base_im = make.image(data=data, title='Base image',
                                         colormap='gray',
                                         interpolation='nearest')
        self.parent.plot.add_item(self.parent.base_im, z=0)
        self.parent.base_im.set_selectable(False)
        if self.parent.mode == 'align':
            row_bounds = np.where(np.diff(np.all(data == 0, axis=1)))[0]
            min_y = row_bounds[0] + 1
            max_y = row_bounds[1] + 1
            y_size = max_y - min_y
            column_bounds = np.where(np.diff(np.all(data == 0, axis=0)))[0]
            min_x = column_bounds[0] + 1
            max_x = column_bounds[-1] + 1
            x_size = max_x - min_x
            # Keep fixed aspect ratio
            image_aspect_ratio = x_size / float(y_size)
            size = self.parent.plot.size()
            window_aspect_ratio = size.width() / float(size.height())
            if image_aspect_ratio > window_aspect_ratio:
                new_y_size = x_size / window_aspect_ratio
                y_padding = (new_y_size - y_size) / 2.0
                min_y -= y_padding
                max_y += y_padding
            else:
                new_x_size = y_size * window_aspect_ratio
                x_padding = (new_x_size - x_size) / 2.0
                min_x -= x_padding
                max_x += x_padding
            self.parent.plot.set_axis_limits(0, max_y, min_y)
            self.parent.plot.set_axis_limits(2, min_x, max_x)

    def transform_rois(self, target_tSeries=None):
        """Returns self.roi_list (a list of Polygon objects) in the target
        space. It just changes the points attribute of the polygons

        Parameters
        ----------
        target_tSeries : UI_tSeries
            Transform rois in self to aligned space of UI_tSeries

        """

        if len(self.roi_list):
            if target_tSeries is None:
                for polygon in self.roi_list:
                    polygon.set_points(polygon.coords[0][:, :2])
            else:
                transform = self.transform(target_tSeries)
                for polygon in self.roi_list:
                    z = int(polygon.coords[0][0, 2])
                    orig_verts = polygon.coords[0][:, :2]
                    new_verts = transform[z](orig_verts)
                    polygon.set_points(new_verts)

    def transform(self, target_tSeries):
        """Return the affine transformation from self to target_tSeries

        Parameters
        ----------
        target_tSeries : UI_tSeries
            The target UI_tSeries to transform the current UI_tSeries into

        Returns
        -------
        array
            2x3 affine transform array

        """

        if target_tSeries not in self.transforms:

            ref_active_channel = self.dataset.channel_names.index(
                self.active_channel)

            target_active_channel = \
                target_tSeries.dataset.channel_names.index(
                    target_tSeries.active_channel)

            self.transforms[target_tSeries] = []

            if '_REGISTRATION_ANCHORS' in self.dataset.ROIs and \
                    '_REGISTRATION_ANCHORS' in target_tSeries.dataset.ROIs:

                ANCHORS = '_REGISTRATION_ANCHORS'

                assert len(
                    self.dataset.ROIs[ANCHORS]) == \
                    self.dataset.frame_shape[0]
                assert len(
                    target_tSeries.dataset.ROIs[ANCHORS]) == \
                    target_tSeries.dataset.frame_shape[0]
                for plane in xrange(self.num_planes):
                    for roi in target_tSeries.dataset.ROIs[ANCHORS]:
                        if roi.coords[0][0, 2] == plane:
                            trg_coords = roi.coords[0][:, :2]
                        else:
                            pass
                    for roi in self.dataset.ROIs[ANCHORS]:
                        if roi.coords[0][0, 2] == plane:
                            src_coords = roi.coords[0][:, :2]
                    assert len(src_coords) == len(trg_coords)

                    mean_dists = []
                    for shift in range(len(src_coords)):
                        points1 = src_coords
                        points2 = np.roll(trg_coords, shift, axis=0)
                        mean_dists.append(
                            np.sum([np.sqrt(np.sum((p1 - p2) ** 2))
                                    for p1, p2 in zip(points1, points2)]))
                    trg_coords = np.roll(
                        trg_coords, np.argmin(mean_dists), axis=0)
                    src_coords = np.vstack(
                        (src_coords, [[0, 0], [0,
                                      self.dataset.frame_shape[1]],
                                      [self.dataset.frame_shape[2], 0],
                                      [self.dataset.frame_shape[2],
                                       self.dataset.frame_shape[1]]]))
                    trg_coords = np.vstack(
                        (trg_coords,
                         [[0, 0], [0, target_tSeries.dataset.frame_shape[1]],
                          [target_tSeries.dataset.frame_shape[2], 0],
                          [target_tSeries.dataset.frame_shape[2],
                           target_tSeries.dataset.frame_shape[1]]]))

                    transform = estimate_coordinate_transform(
                        src_coords, trg_coords, 'piecewise-affine')
                    translation = tf.AffineTransform(
                        translation=target_tSeries.shape[::-1])
                    # translate into same space
                    for tri in range(len(transform.affines)):
                        transform.affines[tri] += translation
                    self.transforms[target_tSeries].append(transform)
            else:
                for plane in xrange(self.num_planes):
                    ref = self.dataset.time_averages[
                        plane, :, :, ref_active_channel]
                    target = target_tSeries.dataset.time_averages[
                        plane, :, :, target_active_channel]

                    transform = estimate_array_transform(
                        ref, target, method='affine')
                    # translate into same space
                    transform += tf.AffineTransform(
                        translation=target_tSeries.shape[::-1])
                    self.transforms[target_tSeries].append(transform)
        # index the result by plane
        return self.transforms[target_tSeries]

    def initialize_rois(self):
        """Load the ROIs and store the original vertices as an
        attribute of the polygon

        """

        self.roi_list = []
        try:
            rois = self.dataset.ROIs[self.active_rois]
        except KeyError:
            return
        else:
            multi_rois = []
            for roi in rois:
                new_rois = []
                for poly in roi.coords:
                    new_roi = UI_ROI(parent=self,
                                     points=poly[:, :2].tolist(),
                                     id=roi.id,
                                     tags=roi.tags,
                                     label=roi.label)
                    new_roi.polygons = poly
                    new_rois.append(new_roi)

                self.roi_list.extend(new_rois)
                # Keep track of ROIs that are MulitPolygons so we make sure
                # that they have the same label later
                if len(new_rois) > 1:
                    multi_rois.append(new_rois)

        # Go through all ROIs and give them labels if they don't have any
        for rois in multi_rois:
            if rois[0].label is None:
                label = next_int([r.label for r in self.roi_list])
                for roi in rois:
                    roi.label = label
                    roi.update_name()

        for roi in self.roi_list:
            if roi.label is None:
                roi.label = next_int([r.label for r in self.roi_list])
                roi.update_name()
                roi.update_color()

    def next_label(self):
        """Return the next un-used label"""
        all_labels = [roi.label for roi in self.parent.plot.get_items() if
                      hasattr(roi, 'label')]
        return next_int(all_labels)

    def update_rois(self, keep_list=None):
        """Update the self.roi_list with the polygons currently on the plot

        keep_list is optionally a list of rois to keep track of
        If no items in keep_list, returns the same list, otherwise returns the
        list with old items replaced by their new roi.
        """

        if keep_list is None:
            keep_list = []

        new_keep_list = []

        # This should only be performed in edit mode, since in align mode all
        # the polygons will be in the wrong space
        assert(self.parent.mode == 'edit')

        # This line is necessary if the user failed to finalize the polygon
        self.parent.freeform_tool.shape = None

        self.roi_list = [r for r in self.roi_list if r.coords[0][0, 2]
                         != self.active_plane]
        # Note need to iterate backwards because convert_polygon modifies
        # plot items list
        for item in reversed(self.parent.plot.get_items()):
            if isinstance(item, UI_ROI):
                if item.coords[0][0, 2] != self.active_plane:
                    continue
                if item.parent == self:
                    item.update_points()
                    self.roi_list.append(item)
                    if item in keep_list:
                        new_keep_list.append(item)
            elif isinstance(item, PolygonShape):
                if item.__class__ == EllipseShape:
                    center = item.get_center()
                    p = item.get_points()
                    radius = np.amax((np.linalg.norm(p[1] - p[0]),
                                      np.linalg.norm(p[1] - p[2]),
                                      np.linalg.norm(p[1] - p[3]))) / 2

                    mask = np.zeros(self.parent.base_im.data.shape, dtype=bool)
                    for x in np.arange(
                            np.floor(center[0] - radius),
                            np.ceil(center[0] + radius)).astype(int):
                        for y in np.arange(
                                np.floor(center[1] - radius),
                                np.ceil(center[1] + radius)).astype(int):

                            d = np.linalg.norm(
                                np.array((x, y)) - np.array(center))

                            if d < radius:
                                mask[y, x] = True

                    poly = mask2poly(mask)

                    points = np.array(poly[0].exterior.coords)[:, :2]
                    new_roi = UI_ROI(self, points, id=None,
                                     label=self.next_label(), tags=None)
                    self.parent.plot.del_item(item)
                    self.parent.plot.add_item(new_roi)
                else:
                    new_roi = UI_ROI.convert_polygon(
                        parent=self, polygon=item)
                coords = new_roi.coords[0]
                coords[:, 2] = self.active_plane
                new_roi.polygons = coords

                if new_roi is not None:
                    self.roi_list.append(new_roi)
                    if item in keep_list:
                        new_keep_list.append(new_roi)
        return new_keep_list

    def save_rois(self, roi_set_name):
        """Save ROIList object"""

        if roi_set_name not in self.roi_sets:
            self.roi_sets.append(roi_set_name)
        self.active_rois = roi_set_name

        # Group ROIs with the same label into the same ROI object
        rois_by_label = {}

        for roi in self.roi_list:
            assert roi.parent == self
            name = roi.label
            verts = roi.coords

            if name in rois_by_label:
                # If two ROIs have the same name they'll have the same
                # tags and id, so no need to check here
                # TODO: ensure that this is true
                rois_by_label[name]['polygons'].extend(verts)
            else:
                rois_by_label[name] = {}
                rois_by_label[name]['tags'] = roi.tags
                rois_by_label[name]['id'] = roi.id
                rois_by_label[name]['label'] = roi.label
                rois_by_label[name]['polygons'] = verts

        ROIs = ROIList([])
        for label in rois_by_label:
            ROIs.append(ROI(polygons=rois_by_label[label]['polygons'],
                            im_shape=self.shape,
                            tags=rois_by_label[label]['tags'],
                            id=rois_by_label[label]['id'],
                            label=rois_by_label[label]['label']))

        self.dataset.add_ROIs(ROIs, label=roi_set_name)


class UI_ROI(PolygonShape, ROI):
    """Class used both to draw polygons as well as store id/tag information
    Always a single polygon.

    """

    def __init__(self, parent, points, id=None, tags=None, label=None):
        PolygonShape.__init__(self, points=points, closed=True)
        ROI.__init__(self, polygons=points, id=id, tags=tags, label=label)
        self.parent = parent

        self.initialize_style()
        self.update_name()

    @staticmethod
    def convert_polygon(polygon, parent):
        """Takes a polygon and returns a similar UI_ROI object."""
        points = polygon.get_points()

        # Make sure the polygon has at least 3 points
        if points.shape[0] <= 2:
            parent.parent.plot.del_item(polygon)
            return None

        new_roi = UI_ROI(parent=parent, points=points.tolist(), id=None,
                         label=parent.next_label(), tags=None)

        coords = new_roi.coords
        coords[0][:, 2] = parent.active_plane
        new_roi.polygons = coords

        parent.parent.plot.del_item(polygon)
        parent.parent.plot.add_item(new_roi)

        return new_roi

    def initialize_style(self):
        self.pen.setWidth(2)
        self.sel_pen.setColor(QColor('yellow'))
        self.sel_pen.setWidth(4)
        self.update_color()

    def update_color(self):
        """Updates the color of the ROI from the colors_dict or a new
        random color

        ROIs are colored first by id, then by label, then randomly

        """

        if self.id is None:
            if self.label is None:
                color = random_color()
            elif self.label in self.parent.parent.colors_dict:
                color = self.parent.parent.colors_dict[self.label]
            else:
                color = random_color()
                self.parent.parent.colors_dict[self.label] = color
        elif self.id in self.parent.parent.colors_dict:
            color = self.parent.parent.colors_dict[self.id]
        else:
            color = random_color()
            self.parent.parent.colors_dict[self.id] = color
        self.pen.setColor(color)

    def update_name(self):
        name = self.label + ':' if self.label is not None else ':'
        for tag in self.tags:
            name += ' ' + tag + ','
        name = name.rstrip(',')
        self.setTitle(name)

    def update_points(self):
        points = self.get_points()
        z = np.empty((len(points), 1))
        z.fill(self.parent.active_plane)
        self.polygons = np.hstack((points, z))

    def toggle_editing(self, value):
        """Lock or unlock the ROIs for editing

        Parameters
        ----------
        value : boolean
            True/False to enable/disable ROIs for editing

        """
        self.set_resizable(value)
        self.set_movable(value)
        self.set_rotatable(value)
        self.set_private(not value)
        self.set_selectable(True)

    def show(self, show_in_list=None):
        if self not in self.parent.parent.plot.items:
            self.parent.parent.plot.add_item(self)
        self.setVisible(True)
        if show_in_list is None:
            # Don't change the list state
            pass
        else:
            self.set_private(not show_in_list)

    def hide(self, show_in_list=None):
        self.setVisible(False)
        if show_in_list is None:
            # Don't change the list state
            pass
        else:
            self.set_private(not show_in_list)


class lockROIsWidget(QDialog):
    def __init__(self, parent, tSeries_list):
        QDialog.__init__(self)

        self.layout = QVBoxLayout()
        self.select_all_button = self.initialize_select_all()

        self.checks = {}
        for tSeries in tSeries_list:
            text = tSeries.dataset.savedir.split('/')[-3:-1]
            c = QCheckBox('{}'.format(text))
            self.layout.addWidget(c)
            self.checks[tSeries] = c

        self.accept_button = QPushButton("Accept", self)
        self.cancel_button = QPushButton("Cancel", self)

        self.layout.addWidget(self.accept_button)
        self.layout.addWidget(self.cancel_button)

        self.accept_button.clicked.connect(
            lambda: self.toggle_lock_status(tSeries_list))

        self.cancel_button.clicked.connect(
            self.cancel)

        self.setLayout(self.layout)
        self.setWindowTitle(QString('Lock ROI IDs during alignment?'))

    def initialize_select_all(self):
        c = QCheckBox('Select All')
        c.clicked.connect(self.select_all)
        self.layout.addWidget(c)

        hline = QFrame()
        hline.setFrameStyle(QFrame.HLine)
        hline.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.layout.addWidget(hline)

        return c

    def select_all(self):
        check_state = self.select_all_button.isChecked()

        for tSeries_checkbox in self.checks.itervalues():
            tSeries_checkbox.setChecked(check_state)

    def toggle_lock_status(self, tSeries_list):
        for tSeries in tSeries_list:
            tSeries.roi_id_lock = self.checks[tSeries].isChecked()
        self.accept()

    def cancel(self):
        self.reject()


class ImportROIsWidget(QDialog, Ui_importROIsWidget):
    """Instance of the ROI Buddy Qt interface."""
    def __init__(self, parent=None):
        """
        Initialize the application
        """
        QDialog.__init__(self)
        self.setupUi(self)
        self.setWindowTitle('Import ROIs')

        self.parent = parent

        # initialize source imaging datasets
        self.initialize_form()

    def initialize_form(self):
        active_dataset = self.parent.tSeries_list.currentItem()
        self.source_datasets = [self.parent.tSeries_list.item(i) for i in
                                range(self.parent.tSeries_list.count())]
        self.source_datasets.remove(active_dataset)

        self.sourceDataset.addItems([QString(x.dataset.savedir) for x in
                                     self.source_datasets])

        target_channels = active_dataset.dataset.channel_names
        self.targetChannel.addItems([QString(x) for x in target_channels])

        self.sourceDataset.currentIndexChanged.connect(
            self.initialize_source_options)

        self.auto_manual.addItems([QString('Auto'), QString('Manual')])
        self.auto_manual.setCurrentIndex(0)

        self.registrationMethod.addItems([QString('affine'),
                                          QString('polynomial'),
                                          QString('piecewise-affine'),
                                          QString('projective'),
                                          QString('similarity')])
        self.registrationMethod.setCurrentIndex(1)

        self.polynomialOrder.setText(QString('2'))

        self.acceptButton.clicked.connect(self.accept)
        self.cancelButton.clicked.connect(self.reject)
        self.sourceDataset.setCurrentIndex(0)
        self.initialize_source_options()

    def initialize_source_options(self):
        self.source_dataset = self.source_datasets[
            self.sourceDataset.currentIndex()]

        source_channels = self.source_dataset.dataset.channel_names
        self.sourceChannel.clear()
        self.sourceChannel.addItems([QString(x) for x in source_channels])

        source_labels = self.source_dataset.dataset.ROIs.keys()
        self.sourceLabel.clear()
        self.sourceLabel.addItems([QString(x) for x in source_labels])

    @staticmethod
    def getParams(parent=None):
        dialog = ImportROIsWidget(parent)
        result = dialog.exec_()

        source_dataset = dialog.source_dataset
        source_channel = str(dialog.sourceChannel.itemText(
            dialog.sourceChannel.currentIndex()))
        source_label = str(dialog.sourceLabel.itemText(
            dialog.sourceLabel.currentIndex()))
        target_channel = str(dialog.targetChannel.itemText(
            dialog.targetChannel.currentIndex()))
        target_label = str(dialog.targetLabel.text())
        copy_properties = dialog.copyRoiProperties.isChecked()

        auto_manual = str(dialog.auto_manual.itemText(
            dialog.auto_manual.currentIndex()))
        reg_method = str(dialog.registrationMethod.itemText(
            dialog.registrationMethod.currentIndex()))
        poly_order = int(dialog.polynomialOrder.text())

        return \
            source_dataset, \
            source_channel, \
            target_channel, \
            source_label, \
            target_label, \
            copy_properties, \
            auto_manual, \
            reg_method, \
            poly_order, \
            result == QDialog.Accepted


def next_int(sequence):
    """Returns the lowest non-negative integer not found in 'sequence'"""
    sequence_ints = []
    for val in sequence:
        try:
            sequence_ints.append(int(val))
        except (ValueError, TypeError):
            pass
    unique_ints = list(set(sequence_ints))
    if len(unique_ints) == 0:
        return 0
    for x in xrange(np.amax(unique_ints)):
        if x not in unique_ints:
            return x
    return np.amax(unique_ints) + 1


def random_color():
    return QColor(qRgb(np.random.randint(255),
                       np.random.randint(255),
                       np.random.randint(255)))


def main():
    app = QApplication(sys.argv)
    form = RoiBuddy()
    form.show()
    app.exec_()

if __name__ == "__main__":
    main()

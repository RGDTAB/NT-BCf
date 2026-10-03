import sys
import slangpy as spy
from enum import Enum
from typing import Sequence
from pathlib import Path

# SlangPy doesn't include a file dialog for linux,
# so this acts as an ImGUI based stand-in
class EmulatedFileDialog():
    # Static member, lets us reopen the file dialog to the same directory
    last_dir = None

    class FileDialogMode(Enum):
        OPEN = 0
        SAVE = 1
        FOLDER = 2

    def __init__(self, parent: spy.ui.Widget, filters: Sequence[spy.platform.FileDialogFilter], mode: FileDialogMode):
        self.mode = mode
        self.filters = filters

        # Select window title
        match mode:
            case self.FileDialogMode.OPEN:
                window_title = 'Open Files'
                confirm_button_label = 'Select File(s)'
            case self.FileDialogMode.SAVE:
                window_title = 'Save File'
                confirm_button_label = 'Save'
            case self.FileDialogMode.FOLDER:
                window_title = 'Open Folder'
                confirm_button_label = 'Select Folder'

        # Open window
        self.window = spy.ui.Window(parent, window_title)

        self.needs_rerender = True # True when entering a new directory
        self.complete = False # True when canceled or path is finished successfully
        self.result = None # Is not None when a path has been selected successfully

        # Toggles the showing of hidden files
        def show_hidden_callback(x):
            self.show_hidden = x
            self.needs_rerender = True
        self.show_hidden = False
        self.show_hidden_checkbox = spy.ui.CheckBox(self.window, 'Show Hidden', False, show_hidden_callback)

        # Group widget that contains all of the files in a directory
        self.browser_group = spy.ui.Group(self.window, 'Browser')
        self.address_bar = spy.ui.InputText(self.window, 'Files', '')
        if mode != self.FileDialogMode.SAVE:
            self.address_bar.enabled = False

        # Open mode allows selecting multiple files at once
        if self.mode == self.FileDialogMode.OPEN:
            self.result = []

        # Close the file dialog and set the result
        def select_callback():
            if self.mode == self.FileDialogMode.OPEN and len(self.result) == 0:
                return

            self.complete = True
            match self.mode:
                case self.FileDialogMode.SAVE:
                    self.result = self.directory.joinpath(self.address_bar.value)
                case self.FileDialogMode.FOLDER:
                    self.result = self.directory
            self.window.close()
        self.select_button = spy.ui.Button(self.window, confirm_button_label, select_callback)

        # Close the file dialog without returning anything
        def cancel_callback():
            self.complete = True
            self.result = None
            self.window.close()
        self.cancel_button = spy.ui.Button(self.window, 'Cancel', cancel_callback)

        self.subdirectories = [] # Directory UI items
        self.directory_files = [] # File UI items

        # cd to home, or last directory if there is one
        if EmulatedFileDialog.last_dir is None:
            self.cd(spy.platform.home_directory())
        else:
            self.cd(EmulatedFileDialog.last_dir)

    # Checks a file against the filters set at initialization
    def matches_filter(self, file: Path) -> bool:
        if len(self.filters) == 0:
            return True

        for dialog_filter in self.filters:
            filter_suffixes = dialog_filter.pattern.replace('*','').split(',')
            if file.suffix in filter_suffixes:
                return True
        return False

    # Updates the file browser
    def render(self):
        if not self.complete:
            self.window.show()

        if not self.needs_rerender:
            return

        # Clear out the previous directory's data
        self.browser_group.remove_all_children()
        self.subdirectories.clear()
        self.directory_files.clear()

        # Parent directory button
        self.subdirectories.append(
            spy.ui.Button(
                self.browser_group,
                '..',
                lambda: self.cd(self.directory.parent),
            )
        )

        # Directories are placed first
        for child in self.directory.iterdir():
            # Is there any other way to detect hidden files/dirs?
            if child.name.startswith('.') and not self.show_hidden:
                continue
            if not child.is_dir():
                continue

            # Add directory, with self.cd function set as the callback
            self.subdirectories.append(
                spy.ui.Button(
                    self.browser_group,
                    child.name,
                    lambda x = child: self.cd(x),
                )
            )

        # Don't show files if we are only interested in folders
        if self.mode == self.FileDialogMode.FOLDER:
            return

        # Next come the files
        for child in self.directory.iterdir():
            # Is there any other way to detect hidden files/dirs?
            if child.name.startswith('.') and not self.show_hidden:
                continue
            if not child.is_file():
                continue

            # Check file against filter
            if not self.matches_filter(child):
                continue

            # For open mode - add/remove this file from our result
            def file_checkbox_callback(value, x = child):
                if value:
                    self.result.append(x)
                else:
                    self.result.remove(x)

                self.address_bar.value = ''
                for path in self.result:
                    self.address_bar.value += f"'{path.name}', "

            if self.mode == self.FileDialogMode.OPEN:
                self.directory_files.append(
                    spy.ui.CheckBox(
                        self.browser_group,
                        child.name,
                        False,
                        file_checkbox_callback,
                    )
                )
            else:
                # Just a simple text entry for existing files when in save mode
                self.directory_files.append(
                    spy.ui.Text(
                        self.browser_group,
                        child.name,
                    )
                )

        self.needs_rerender = False

    # Change directory to directory arg
    def cd(self, directory: Path):
        # No UI updates here - that happens in self.render()
        self.directory = directory
        EmulatedFileDialog.last_dir = directory
        self.browser_group.label = str(directory)
        self.needs_rerender = True


# Native open file dialog with emulated one as a fallback
# Difference between them is that the emulated file dialog can return multiple files
def open_file_dialog(parent: spy.ui.Widget, filters: Sequence[spy.platform.FileDialogFilter]):
    if sys.platform != 'linux':
        return spy.platform.open_file_dialog(filters)
    else:
        return EmulatedFileDialog(parent, filters, EmulatedFileDialog.FileDialogMode.OPEN)

# Native save file dialog with emulated one as a fallback
def save_file_dialog(parent: spy.ui.Widget, filters: Sequence[spy.platform.FileDialogFilter]):
    if sys.platform != 'linux':
        return spy.platform.save_file_dialog(filters)
    else:
        return EmulatedFileDialog(parent, filters, EmulatedFileDialog.FileDialogMode.SAVE)

# Native choose file dialog with emulated one as a fallback
# Currently not using this
def choose_folder_dialog(parent: spy.ui.Widget):
    if sys.platform != 'linux':
        return spy.platform.choose_folder_dialog(filters)
    else:
        return EmulatedFileDialog(parent, filters, EmulatedFileDialog.FileDialogMode.FOLDER)


import types

from PyQt6.QtWidgets import QDialog, QHBoxLayout, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QFormLayout, QGroupBox, \
    QMessageBox
from VisualInterface.form_ui import Ui_Form
from VisualInterface.utils import *
import sys


def validate_list_from_string(s, to_type):
    try:
        print(s, to_type)
        res = list(map(to_type, s.split(",")))
        if len(res) == 0:
            raise ValueError()
        print(res, s.split(","))
        return True
    except ValueError:
        return False


class ListLineEdit(QLineEdit):
    def __init__(self, data_type):
        super().__init__()
        self.default_style = self.styleSheet()
        self.data_type = data_type

    def focusOutEvent(self, event):
        if not validate_list_from_string(self.text(), self.data_type):
            self.setStyleSheet("QLineEdit{border: 1px solid red}")
        else:
            self.setStyleSheet(self.default_style)
        super().focusOutEvent(event)


class Form(QDialog, Ui_Form):
    def __init__(self, form_type):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Add " + form_type)
        self.form_layout = QFormLayout(parent=self.scroll_area)
        self.form_groupbox = QGroupBox()
        self.form_groupbox.setLayout(self.form_layout)
        self.scroll_area.setWidget(self.form_groupbox)
        if form_type == "trader":
            self.type_box.addItems(get_agents_list())
        elif form_type == "event":
            self.type_box.addItems(get_events_list())
        elif form_type == "broker":
            self.type_box.addItems(get_brokers_list())
        elif form_type == "stats":
            self.type_box.addItems(get_stats_list())
        else:
            print("Invalid")
            sys.exit(1)
        self.type_box.currentTextChanged.connect(self.sync_form)
        self.form_type = form_type
        self.class_type = "agents"
        if self.form_type == "event":
            self.class_type = "events"
        elif self.form_type == "stats":
            self.class_type = "stats"
        self.sync_form()

    def clear_form(self):
        while self.form_layout.count():
            wid = self.form_layout.takeAt(0)
            if wid.widget():
                wid.widget().deleteLater()

    def accept(self):
        for i in range(self.form_layout.count() // 2):
            param_name = self.form_layout.itemAt(i * 2).widget().text()
            type_hint = param_name[param_name.find("(") + 1:-1]
            ans = self.form_layout.itemAt(i * 2 + 1).widget().text()
            param = param_name[:param_name.find("(")]
            if "Optional" in type_hint and ans == "":
                continue
            if not can_bruteforce(self.class_type, self.type_box.currentText(),
                                  param) and "," in ans and "List" not in type_hint:
                QMessageBox.about(self, "Error", f"Invalid input line {i + 1}")
                return
            if type_hint == "int" or type_hint == "List[int]" or type_hint == "Optional[int]" or \
                    type_hint == "Optional[List[int]]":
                if not validate_list_from_string(ans, int):
                    QMessageBox.about(self, "Error", f"Invalid input line {i + 1}")
                    return
            elif type_hint == "float" or type_hint == "float or int":
                if not validate_list_from_string(ans, float):
                    QMessageBox.about(self, "Error", f"Invalid input line {i + 1}")
                    return

        super().accept()

    def sync_form(self):
        self.clear_form()
        item_name = self.type_box.currentText()
        di = get_arg_type_dict(self.class_type, item_name)
        for field in get_fields(self.class_type, item_name):
            type_hint = ""
            tool_tip = get_parameter_info(self.class_type, item_name, field)
            if di[field] in [int, float]:
                type_hint = di[field].__name__
            else:
                type_hint = str(di[field]).replace("typing.", "")
            if "List" in type_hint:
                tool_tip += "(comma-separated values)"
            if type_hint == "_empty":
                type_hint = "no type specified"
            if "market" in field:
                type_hint = "List[int]"
            type_hint = type_hint.replace("|", "or")
            if self.class_type == "stats" and field == "info":
                type_hint = "int"
                tool_tip = "id of simulator info"
            label = QLabel(field + f"({type_hint})")
            if can_bruteforce(self.class_type, item_name, field):
                tool_tip += "(can bruteforce)"
            label.setToolTip(tool_tip)
            line_edit = QLineEdit()
            if type_hint == "int" or type_hint == "List[int]":
                line_edit = ListLineEdit(int)
            elif type_hint == "float or int" or type_hint == "float":
                line_edit = ListLineEdit(float)
            def_val = get_default_value(self.class_type, item_name, field)
            if def_val is not None:
                line_edit.setText(str(def_val))
            if field == "count":
                line_edit.setText("1")
            self.form_layout.addRow(label, line_edit)
        sb_count = ListLineEdit(int)
        if self.class_type == "agents":
            self.form_layout.addRow(QLabel("count(int)"), sb_count)

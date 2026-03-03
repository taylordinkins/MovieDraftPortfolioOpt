from __future__ import annotations

import os
import sys
import traceback

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6 import QtCore, QtWidgets

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import calculations
import storage
from scraper import scrape_hsx, scrape_price_history, scrape_release_dates


def _clean_name(name, ticker):
    if not isinstance(name, str) or not name:
        return "(not found in HSX)"
    prefix = f"{ticker}: "
    return name[len(prefix):] if name.startswith(prefix) else name


def _fmt_money(value) -> str:
    try:
        num = float(value)
    except Exception:
        return "-"
    if np.isnan(num):
        return "-"
    return f"${num:.2f}"


class DataFrameTableModel(QtCore.QAbstractTableModel):
    def __init__(self, df: pd.DataFrame | None = None):
        super().__init__()
        self._df = df.copy() if df is not None else pd.DataFrame()

    def set_dataframe(self, df: pd.DataFrame):
        self.beginResetModel()
        self._df = df.copy() if df is not None else pd.DataFrame()
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._df.index)

    def columnCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._df.columns)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        if role not in (QtCore.Qt.DisplayRole, QtCore.Qt.EditRole):
            return None
        value = self._df.iat[index.row(), index.column()]
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return ""
            return f"{value:.4f}" if abs(value) < 1 else f"{value:.2f}"
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d")
        return str(value)

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return None
        if orientation == QtCore.Qt.Horizontal:
            if section < len(self._df.columns):
                return str(self._df.columns[section])
            return ""
        return str(section + 1)


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.figure = Figure(figsize=(8, 4), tight_layout=True)
        self.ax = self.figure.add_subplot(111)
        super().__init__(self.figure)


class DraftToolWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HSX Movie Draft Tool (GUI)")
        self.resize(1500, 930)

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        self._build_dashboard_tab()
        self._build_draft_state_tab()
        self._build_history_tab()
        self._build_data_tab()

        self.refresh_all()

    def _set_status(self, message: str):
        self.statusBar().showMessage(message, 8000)

    def _show_error(self, message: str, detail: str | None = None):
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Critical)
        box.setWindowTitle("Error")
        box.setText(message)
        if detail:
            box.setDetailedText(detail)
        box.exec()

    def _show_info(self, message: str):
        QtWidgets.QMessageBox.information(self, "Info", message)

    def _resolve_runtime_seed_for_run(self, settings: dict) -> tuple[int, str]:
        mode = str(settings.get("strategy_mc_seed_mode", "fixed") or "fixed").strip().lower()
        seed = pd.to_numeric(pd.Series([settings.get("strategy_mc_random_seed", 123)]), errors="coerce").fillna(123).iloc[0]
        if mode == "random":
            run_seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
            return run_seed, "random"
        return int(seed), "fixed"

    def refresh_all(self):
        storage.ensure_auction_state_seeded()
        self.refresh_dashboard_inputs()
        self.refresh_dashboard_output()
        self.refresh_draft_state()
        self.refresh_history()
        self.refresh_data_tab()

    # ---------- Dashboard ----------

    def _build_dashboard_tab(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Dashboard")
        root = QtWidgets.QVBoxLayout(tab)

        controls = QtWidgets.QGridLayout()
        root.addLayout(controls)

        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems(["conservative", "balanced", "aggressive"])
        controls.addWidget(QtWidgets.QLabel("Preset"), 0, 0)
        controls.addWidget(self.preset_combo, 0, 1)

        self.budget_mode_combo = QtWidgets.QComboBox()
        self.budget_mode_combo.addItems(["personal", "league", "custom"])
        controls.addWidget(QtWidgets.QLabel("Budget Basis"), 0, 2)
        controls.addWidget(self.budget_mode_combo, 0, 3)

        self.custom_budget_spin = QtWidgets.QDoubleSpinBox()
        self.custom_budget_spin.setRange(0.0, 50000.0)
        self.custom_budget_spin.setDecimals(2)
        self.custom_budget_spin.setValue(200.0)
        controls.addWidget(QtWidgets.QLabel("Custom Budget"), 0, 4)
        controls.addWidget(self.custom_budget_spin, 0, 5)

        self.user_combo = QtWidgets.QComboBox()
        controls.addWidget(QtWidgets.QLabel("Personal User"), 0, 6)
        controls.addWidget(self.user_combo, 0, 7)

        self.risk_combo = QtWidgets.QComboBox()
        self.risk_combo.addItems(["bootstrap", "student_t"])
        controls.addWidget(QtWidgets.QLabel("Risk Model"), 1, 0)
        controls.addWidget(self.risk_combo, 1, 1)

        self.objective_combo = QtWidgets.QComboBox()
        self.objective_combo.addItems(["expected_gross", "win_probability"])
        controls.addWidget(QtWidgets.QLabel("Objective"), 1, 2)
        controls.addWidget(self.objective_combo, 1, 3)

        self.cost_combo = QtWidgets.QComboBox()
        self.cost_combo.addItems(["target_bid", "market_fair_bid", "current_price"])
        controls.addWidget(QtWidgets.QLabel("Optimizer Cost"), 1, 4)
        controls.addWidget(self.cost_combo, 1, 5)

        self.integer_check = QtWidgets.QCheckBox("Integer Bids")
        controls.addWidget(self.integer_check, 1, 6)
        self.prev_bid_spin = QtWidgets.QSpinBox()
        self.prev_bid_spin.setRange(0, 100000)
        controls.addWidget(QtWidgets.QLabel("Previous Bid"), 1, 7)
        controls.addWidget(self.prev_bid_spin, 1, 8)

        self.dashboard_context_label = QtWidgets.QLabel("Draft context: -")
        controls.addWidget(self.dashboard_context_label, 2, 0, 1, 6)

        run_btn = QtWidgets.QPushButton("Run Dashboard")
        run_btn.clicked.connect(self.run_dashboard)
        controls.addWidget(run_btn, 2, 7, 1, 2)

        # Advanced knobs (high-impact tuning controls).
        controls.addWidget(QtWidgets.QLabel("Bid Mult"), 3, 0)
        self.bid_mult_spin = QtWidgets.QDoubleSpinBox()
        self.bid_mult_spin.setRange(0.50, 2.00)
        self.bid_mult_spin.setDecimals(2)
        self.bid_mult_spin.setSingleStep(0.05)
        controls.addWidget(self.bid_mult_spin, 3, 1)

        controls.addWidget(QtWidgets.QLabel("Base Cap"), 3, 2)
        self.base_cap_spin = QtWidgets.QDoubleSpinBox()
        self.base_cap_spin.setRange(0.01, 1.00)
        self.base_cap_spin.setDecimals(2)
        self.base_cap_spin.setSingleStep(0.01)
        controls.addWidget(self.base_cap_spin, 3, 3)

        controls.addWidget(QtWidgets.QLabel("Fair Stress Cap"), 3, 4)
        self.fair_cap_spin = QtWidgets.QDoubleSpinBox()
        self.fair_cap_spin.setRange(0.01, 1.00)
        self.fair_cap_spin.setDecimals(2)
        self.fair_cap_spin.setSingleStep(0.01)
        controls.addWidget(self.fair_cap_spin, 3, 5)

        self.quality_filter_check = QtWidgets.QCheckBox("Quality Filter")
        controls.addWidget(self.quality_filter_check, 3, 6)

        controls.addWidget(QtWidgets.QLabel("Min P(Edge)"), 3, 7)
        self.min_edge_spin = QtWidgets.QDoubleSpinBox()
        self.min_edge_spin.setRange(0.0, 1.0)
        self.min_edge_spin.setDecimals(2)
        self.min_edge_spin.setSingleStep(0.05)
        controls.addWidget(self.min_edge_spin, 3, 8)

        controls.addWidget(QtWidgets.QLabel("Max P(DD)"), 4, 0)
        self.max_dd_spin = QtWidgets.QDoubleSpinBox()
        self.max_dd_spin.setRange(0.0, 1.0)
        self.max_dd_spin.setDecimals(2)
        self.max_dd_spin.setSingleStep(0.05)
        controls.addWidget(self.max_dd_spin, 4, 1)

        controls.addWidget(QtWidgets.QLabel("DD Threshold"), 4, 2)
        self.dd_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.dd_threshold_spin.setRange(0.0, 1.0)
        self.dd_threshold_spin.setDecimals(2)
        self.dd_threshold_spin.setSingleStep(0.05)
        controls.addWidget(self.dd_threshold_spin, 4, 3)

        controls.addWidget(QtWidgets.QLabel("MC Samples"), 4, 4)
        self.mc_samples_spin = QtWidgets.QSpinBox()
        self.mc_samples_spin.setRange(100, 20000)
        controls.addWidget(self.mc_samples_spin, 4, 5)

        controls.addWidget(QtWidgets.QLabel("Opponents"), 4, 6)
        self.mc_opp_spin = QtWidgets.QSpinBox()
        self.mc_opp_spin.setRange(1, 30)
        controls.addWidget(self.mc_opp_spin, 4, 7)

        controls.addWidget(QtWidgets.QLabel("MC Candidates"), 4, 8)
        self.mc_candidates_spin = QtWidgets.QSpinBox()
        self.mc_candidates_spin.setRange(20, 2000)
        controls.addWidget(self.mc_candidates_spin, 4, 9)

        controls.addWidget(QtWidgets.QLabel("Risk A(vol)"), 5, 0)
        self.risk_a_spin = QtWidgets.QDoubleSpinBox()
        self.risk_a_spin.setRange(0.0, 10.0)
        self.risk_a_spin.setDecimals(2)
        self.risk_a_spin.setSingleStep(0.1)
        controls.addWidget(self.risk_a_spin, 5, 1)

        controls.addWidget(QtWidgets.QLabel("Risk B(DD)"), 5, 2)
        self.risk_b_spin = QtWidgets.QDoubleSpinBox()
        self.risk_b_spin.setRange(0.0, 10.0)
        self.risk_b_spin.setDecimals(2)
        self.risk_b_spin.setSingleStep(0.1)
        controls.addWidget(self.risk_b_spin, 5, 3)

        controls.addWidget(QtWidgets.QLabel("Risk C(release)"), 5, 4)
        self.risk_c_spin = QtWidgets.QDoubleSpinBox()
        self.risk_c_spin.setRange(0.0, 5.0)
        self.risk_c_spin.setDecimals(2)
        self.risk_c_spin.setSingleStep(0.05)
        controls.addWidget(self.risk_c_spin, 5, 5)

        controls.addWidget(QtWidgets.QLabel("Risk Max"), 5, 6)
        self.risk_max_spin = QtWidgets.QDoubleSpinBox()
        self.risk_max_spin.setRange(0.0, 0.95)
        self.risk_max_spin.setDecimals(2)
        self.risk_max_spin.setSingleStep(0.05)
        controls.addWidget(self.risk_max_spin, 5, 7)

        controls.addWidget(QtWidgets.QLabel("Risk Window"), 5, 8)
        self.risk_window_spin = QtWidgets.QSpinBox()
        self.risk_window_spin.setRange(1, 180)
        controls.addWidget(self.risk_window_spin, 5, 9)

        controls.addWidget(QtWidgets.QLabel("Div Pen"), 6, 0)
        self.div_pen_spin = QtWidgets.QDoubleSpinBox()
        self.div_pen_spin.setRange(0.0, 1.0)
        self.div_pen_spin.setDecimals(2)
        self.div_pen_spin.setSingleStep(0.05)
        controls.addWidget(self.div_pen_spin, 6, 1)

        controls.addWidget(QtWidgets.QLabel("Corr Pen"), 6, 2)
        self.corr_pen_spin = QtWidgets.QDoubleSpinBox()
        self.corr_pen_spin.setRange(0.0, 1.0)
        self.corr_pen_spin.setDecimals(2)
        self.corr_pen_spin.setSingleStep(0.05)
        controls.addWidget(self.corr_pen_spin, 6, 3)

        controls.addWidget(QtWidgets.QLabel("Prob Samples"), 6, 4)
        self.bootstrap_samples_spin = QtWidgets.QSpinBox()
        self.bootstrap_samples_spin.setRange(100, 20000)
        controls.addWidget(self.bootstrap_samples_spin, 6, 5)

        controls.addWidget(QtWidgets.QLabel("Opp Noise"), 6, 6)
        self.mc_noise_spin = QtWidgets.QDoubleSpinBox()
        self.mc_noise_spin.setRange(0.01, 1.0)
        self.mc_noise_spin.setDecimals(2)
        self.mc_noise_spin.setSingleStep(0.05)
        controls.addWidget(self.mc_noise_spin, 6, 7)

        controls.addWidget(QtWidgets.QLabel("Opp Agg SD"), 6, 8)
        self.mc_aggr_spin = QtWidgets.QDoubleSpinBox()
        self.mc_aggr_spin.setRange(0.0, 1.0)
        self.mc_aggr_spin.setDecimals(2)
        self.mc_aggr_spin.setSingleStep(0.05)
        controls.addWidget(self.mc_aggr_spin, 6, 9)

        controls.addWidget(QtWidgets.QLabel("Conc DD Thresh"), 7, 0)
        self.mc_conc_spin = QtWidgets.QDoubleSpinBox()
        self.mc_conc_spin.setRange(0.0, 1.0)
        self.mc_conc_spin.setDecimals(2)
        self.mc_conc_spin.setSingleStep(0.05)
        controls.addWidget(self.mc_conc_spin, 7, 1)

        controls.addWidget(QtWidgets.QLabel("Seed Mode"), 7, 2)
        self.seed_mode_combo = QtWidgets.QComboBox()
        self.seed_mode_combo.addItems(["fixed", "random"])
        self.seed_mode_combo.currentTextChanged.connect(
            lambda txt: self.seed_spin.setEnabled(str(txt).strip().lower() == "fixed")
        )
        controls.addWidget(self.seed_mode_combo, 7, 3)

        controls.addWidget(QtWidgets.QLabel("Fixed Seed"), 7, 4)
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        controls.addWidget(self.seed_spin, 7, 5)

        controls.addWidget(QtWidgets.QLabel("Corr Mode"), 8, 0)
        self.corr_mode_combo = QtWidgets.QComboBox()
        self.corr_mode_combo.addItems(["independent", "gaussian_copula", "t_copula"])
        controls.addWidget(self.corr_mode_combo, 8, 1)

        controls.addWidget(QtWidgets.QLabel("Corr Shrink"), 8, 2)
        self.corr_shrink_spin = QtWidgets.QDoubleSpinBox()
        self.corr_shrink_spin.setRange(0.0, 1.0)
        self.corr_shrink_spin.setDecimals(2)
        self.corr_shrink_spin.setSingleStep(0.05)
        controls.addWidget(self.corr_shrink_spin, 8, 3)

        controls.addWidget(QtWidgets.QLabel("Corr MinPts"), 8, 4)
        self.corr_min_points_spin = QtWidgets.QSpinBox()
        self.corr_min_points_spin.setRange(5, 400)
        controls.addWidget(self.corr_min_points_spin, 8, 5)

        controls.addWidget(QtWidgets.QLabel("Corr Floor"), 8, 6)
        self.corr_floor_spin = QtWidgets.QDoubleSpinBox()
        self.corr_floor_spin.setRange(1e-9, 0.1)
        self.corr_floor_spin.setDecimals(6)
        self.corr_floor_spin.setSingleStep(1e-4)
        controls.addWidget(self.corr_floor_spin, 8, 7)

        controls.addWidget(QtWidgets.QLabel("Corr t-DF"), 8, 8)
        self.corr_t_df_spin = QtWidgets.QSpinBox()
        self.corr_t_df_spin.setRange(3, 60)
        controls.addWidget(self.corr_t_df_spin, 8, 9)

        controls.addWidget(QtWidgets.QLabel("Search Mode"), 9, 0)
        self.search_mode_combo = QtWidgets.QComboBox()
        self.search_mode_combo.addItems(["current_sampled", "random_multistart", "local_search", "genetic"])
        controls.addWidget(self.search_mode_combo, 9, 1)

        controls.addWidget(QtWidgets.QLabel("Search Cands"), 9, 2)
        self.search_candidates_spin = QtWidgets.QSpinBox()
        self.search_candidates_spin.setRange(20, 5000)
        controls.addWidget(self.search_candidates_spin, 9, 3)

        controls.addWidget(QtWidgets.QLabel("Local Iters"), 9, 4)
        self.search_local_iters_spin = QtWidgets.QSpinBox()
        self.search_local_iters_spin.setRange(0, 500)
        controls.addWidget(self.search_local_iters_spin, 9, 5)

        controls.addWidget(QtWidgets.QLabel("GA Pop"), 9, 6)
        self.search_population_spin = QtWidgets.QSpinBox()
        self.search_population_spin.setRange(20, 500)
        controls.addWidget(self.search_population_spin, 9, 7)

        controls.addWidget(QtWidgets.QLabel("GA Gens"), 9, 8)
        self.search_generations_spin = QtWidgets.QSpinBox()
        self.search_generations_spin.setRange(1, 500)
        controls.addWidget(self.search_generations_spin, 9, 9)

        controls.addWidget(QtWidgets.QLabel("GA Elite"), 10, 0)
        self.search_elite_frac_spin = QtWidgets.QDoubleSpinBox()
        self.search_elite_frac_spin.setRange(0.05, 0.90)
        self.search_elite_frac_spin.setDecimals(2)
        self.search_elite_frac_spin.setSingleStep(0.05)
        controls.addWidget(self.search_elite_frac_spin, 10, 1)

        controls.addWidget(QtWidgets.QLabel("GA Mut"), 10, 2)
        self.search_mutation_rate_spin = QtWidgets.QDoubleSpinBox()
        self.search_mutation_rate_spin.setRange(0.0, 0.80)
        self.search_mutation_rate_spin.setDecimals(2)
        self.search_mutation_rate_spin.setSingleStep(0.02)
        controls.addWidget(self.search_mutation_rate_spin, 10, 3)

        controls.addWidget(QtWidgets.QLabel("Opp Profile"), 10, 4)
        self.opponent_profile_combo = QtWidgets.QComboBox()
        self.opponent_profile_combo.addItems(["passive_value", "balanced_field", "aggressive_bidup"])
        controls.addWidget(self.opponent_profile_combo, 10, 5)

        controls.addWidget(QtWidgets.QLabel("Bid-up"), 10, 6)
        self.opp_bidup_spin = QtWidgets.QDoubleSpinBox()
        self.opp_bidup_spin.setRange(0.0, 1.0)
        self.opp_bidup_spin.setDecimals(2)
        self.opp_bidup_spin.setSingleStep(0.05)
        controls.addWidget(self.opp_bidup_spin, 10, 7)

        controls.addWidget(QtWidgets.QLabel("Cash Conserve"), 10, 8)
        self.opp_cash_conserve_spin = QtWidgets.QDoubleSpinBox()
        self.opp_cash_conserve_spin.setRange(-0.50, 0.80)
        self.opp_cash_conserve_spin.setDecimals(2)
        self.opp_cash_conserve_spin.setSingleStep(0.05)
        controls.addWidget(self.opp_cash_conserve_spin, 10, 9)

        controls.addWidget(QtWidgets.QLabel("Saved Profile"), 11, 0)
        self.profile_combo = QtWidgets.QComboBox()
        controls.addWidget(self.profile_combo, 11, 1, 1, 3)
        load_profile_btn = QtWidgets.QPushButton("Load Profile")
        load_profile_btn.clicked.connect(self.on_load_strategy_profile)
        controls.addWidget(load_profile_btn, 11, 4)
        save_profile_btn = QtWidgets.QPushButton("Save Profile")
        save_profile_btn.clicked.connect(self.on_save_strategy_profile)
        controls.addWidget(save_profile_btn, 11, 5)
        refresh_profiles_btn = QtWidgets.QPushButton("Refresh Profiles")
        refresh_profiles_btn.clicked.connect(lambda: self._refresh_profile_combo(self.profile_combo.currentData() or ""))
        controls.addWidget(refresh_profiles_btn, 11, 6)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        root.addWidget(splitter, 1)

        upper = QtWidgets.QWidget()
        upper_layout = QtWidgets.QVBoxLayout(upper)
        upper_layout.addWidget(QtWidgets.QLabel("Available Movies Ranked"))
        self.dashboard_table = QtWidgets.QTableView()
        self.dashboard_model = DataFrameTableModel(pd.DataFrame())
        self.dashboard_table.setModel(self.dashboard_model)
        self.dashboard_table.setSortingEnabled(True)
        self.dashboard_table.horizontalHeader().setStretchLastSection(False)
        upper_layout.addWidget(self.dashboard_table)
        splitter.addWidget(upper)

        lower = QtWidgets.QWidget()
        lower_layout = QtWidgets.QHBoxLayout(lower)
        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("Selected Portfolio"))
        self.selected_table = QtWidgets.QTableView()
        self.selected_model = DataFrameTableModel(pd.DataFrame())
        self.selected_table.setModel(self.selected_model)
        self.selected_table.horizontalHeader().setStretchLastSection(False)
        left.addWidget(self.selected_table)
        lower_layout.addLayout(left, 2)

        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("Summary + Glossary"))
        self.dashboard_summary = QtWidgets.QPlainTextEdit()
        self.dashboard_summary.setReadOnly(True)
        right.addWidget(self.dashboard_summary)
        lower_layout.addLayout(right, 1)
        splitter.addWidget(lower)
        splitter.setSizes([600, 300])

    def _set_combo_value(self, combo: QtWidgets.QComboBox, value: str):
        idx = combo.findText(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _autosize_table_columns(self, table: QtWidgets.QTableView, min_width: int = 90, max_width: int = 360):
        """Resize columns to fit header/content without over-expanding the final column."""
        model = table.model()
        if model is None:
            return
        header = table.horizontalHeader()
        header.setStretchLastSection(False)
        col_count = model.columnCount()
        for col in range(col_count):
            table.resizeColumnToContents(col)
            width = table.columnWidth(col)
            width = max(min_width, min(width + 16, max_width))
            table.setColumnWidth(col, width)

    def _refresh_profile_combo(self, selected_name: str = ""):
        names = storage.list_strategy_profile_names()
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.addItem("(none)", "")
        for name in names:
            self.profile_combo.addItem(name, name)
        if selected_name:
            idx = self.profile_combo.findData(selected_name)
            if idx >= 0:
                self.profile_combo.setCurrentIndex(idx)
            else:
                self.profile_combo.setCurrentIndex(0)
        else:
            self.profile_combo.setCurrentIndex(0)
        self.profile_combo.blockSignals(False)

    def _apply_dashboard_settings_to_controls(self, settings: dict, people: pd.DataFrame, state: dict):
        self._set_combo_value(self.preset_combo, str(settings.get("strategy_active_preset", "balanced")))
        self._set_combo_value(self.risk_combo, str(settings.get("strategy_risk_model", "bootstrap")))
        self._set_combo_value(self.objective_combo, str(settings.get("strategy_objective", "expected_gross")))
        self._set_combo_value(self.cost_combo, str(settings.get("strategy_optimizer_cost_col", "target_bid")))
        self._set_combo_value(self.budget_mode_combo, str(settings.get("strategy_budget_mode_preference", "personal")))
        self.custom_budget_spin.setValue(float(settings.get("strategy_custom_budget_amount", 200.0)))

        self.integer_check.setChecked(bool(settings.get("strategy_integer_bid_mode", False)))
        saved_prev = int(max(settings.get("strategy_integer_prev_bid", 0), 0))
        state_prev = int(max(state.get("current_bid", 0), 0))
        self.prev_bid_spin.setValue(state_prev if state_prev > 0 else saved_prev)

        self.bid_mult_spin.setValue(float(settings.get("strategy_bid_multiplier", 1.00)))
        self.base_cap_spin.setValue(float(settings.get("strategy_max_budget_pct_per_film", 0.22)))
        self.fair_cap_spin.setValue(float(settings.get("strategy_market_fair_stresstest_cap", 0.55)))
        self.quality_filter_check.setChecked(bool(settings.get("strategy_enable_quality_filters", True)))
        self.min_edge_spin.setValue(float(settings.get("strategy_min_prob_positive_edge", 0.20)))
        self.max_dd_spin.setValue(float(settings.get("strategy_max_prob_large_drawdown", 0.20)))
        self.dd_threshold_spin.setValue(float(settings.get("strategy_drawdown_threshold", 0.15)))
        self.mc_samples_spin.setValue(int(settings.get("strategy_mc_samples", 1500)))
        self.mc_opp_spin.setValue(int(settings.get("strategy_mc_num_opponents", 7)))
        self.mc_candidates_spin.setValue(int(settings.get("strategy_mc_candidate_portfolios", 120)))
        self.risk_a_spin.setValue(float(settings.get("strategy_risk_a_vol", 3.00)))
        self.risk_b_spin.setValue(float(settings.get("strategy_risk_b_drawdown", 1.40)))
        self.risk_c_spin.setValue(float(settings.get("strategy_risk_c_release", 0.45)))
        self.risk_max_spin.setValue(float(settings.get("strategy_risk_max_penalty", 0.45)))
        self.risk_window_spin.setValue(int(settings.get("strategy_risk_release_window_days", 30)))
        self.div_pen_spin.setValue(float(settings.get("strategy_diversification_penalty", 0.00)))
        self.corr_pen_spin.setValue(float(settings.get("strategy_correlation_penalty", 0.00)))
        self.bootstrap_samples_spin.setValue(int(settings.get("strategy_bootstrap_samples", 1000)))
        self.mc_noise_spin.setValue(float(settings.get("strategy_mc_opponent_noise", 0.30)))
        self.mc_aggr_spin.setValue(float(settings.get("strategy_mc_aggression_sd", 0.10)))
        self.mc_conc_spin.setValue(float(settings.get("strategy_mc_concentration_threshold", 0.40)))
        self._set_combo_value(self.seed_mode_combo, str(settings.get("strategy_mc_seed_mode", "fixed")))
        self.seed_spin.setValue(int(settings.get("strategy_mc_random_seed", 123)))
        self.seed_spin.setEnabled(self.seed_mode_combo.currentText().strip().lower() == "fixed")
        self._set_combo_value(self.corr_mode_combo, str(settings.get("strategy_corr_simulation_mode", "independent")))
        self.corr_shrink_spin.setValue(float(settings.get("strategy_corr_shrinkage", 0.20)))
        self.corr_min_points_spin.setValue(int(settings.get("strategy_corr_min_history_points", 20)))
        self.corr_floor_spin.setValue(float(settings.get("strategy_corr_floor", 1e-4)))
        self.corr_t_df_spin.setValue(int(settings.get("strategy_corr_t_df", 8)))
        self._set_combo_value(self.search_mode_combo, str(settings.get("strategy_search_mode", "current_sampled")))
        self.search_candidates_spin.setValue(int(settings.get("strategy_search_candidates", settings.get("strategy_mc_candidate_portfolios", 120))))
        self.search_local_iters_spin.setValue(int(settings.get("strategy_search_local_iters", 25)))
        self.search_population_spin.setValue(int(settings.get("strategy_search_population", 80)))
        self.search_generations_spin.setValue(int(settings.get("strategy_search_generations", 30)))
        self.search_elite_frac_spin.setValue(float(settings.get("strategy_search_elite_frac", 0.20)))
        self.search_mutation_rate_spin.setValue(float(settings.get("strategy_search_mutation_rate", 0.08)))
        self._set_combo_value(self.opponent_profile_combo, str(settings.get("strategy_opponent_profile", "balanced_field")))
        self.opp_bidup_spin.setValue(float(settings.get("strategy_opponent_bidup_strength", 0.00)))
        self.opp_cash_conserve_spin.setValue(float(settings.get("strategy_opponent_cash_conservation", 0.00)))

        cur_movie = str(state.get("current_movie", "") or "").strip().upper()
        self.dashboard_context_label.setText(
            f"Draft context: {cur_movie or '-'}  previous_bid={self.prev_bid_spin.value()}"
        )

        cur_user = str(settings.get("strategy_user_name", "") or "")
        self.user_combo.blockSignals(True)
        self.user_combo.clear()
        self.user_combo.addItem("")
        for n in people.get("name", pd.Series(dtype=object)).astype(str).tolist():
            self.user_combo.addItem(n)
        user_idx = self.user_combo.findText(cur_user)
        if user_idx >= 0:
            self.user_combo.setCurrentIndex(user_idx)
        self.user_combo.blockSignals(False)

    def _collect_dashboard_strategy_settings_from_controls(self) -> dict:
        settings = storage.load_strategy_runtime_defaults()
        settings["strategy_active_preset"] = self.preset_combo.currentText().strip().lower()
        settings["strategy_risk_model"] = self.risk_combo.currentText().strip().lower()
        settings["strategy_objective"] = self.objective_combo.currentText().strip().lower()
        settings["strategy_integer_bid_mode"] = bool(self.integer_check.isChecked())
        settings["strategy_integer_prev_bid"] = int(self.prev_bid_spin.value())
        settings["strategy_budget_mode_preference"] = self.budget_mode_combo.currentText().strip().lower()
        settings["strategy_custom_budget_amount"] = float(self.custom_budget_spin.value())
        settings["strategy_optimizer_cost_col"] = self.cost_combo.currentText().strip()
        settings["strategy_bid_multiplier"] = float(self.bid_mult_spin.value())
        settings["strategy_max_budget_pct_per_film"] = float(self.base_cap_spin.value())
        settings["strategy_market_fair_stresstest_cap"] = float(self.fair_cap_spin.value())
        settings["strategy_enable_quality_filters"] = bool(self.quality_filter_check.isChecked())
        settings["strategy_min_prob_positive_edge"] = float(self.min_edge_spin.value())
        settings["strategy_max_prob_large_drawdown"] = float(self.max_dd_spin.value())
        settings["strategy_drawdown_threshold"] = float(self.dd_threshold_spin.value())
        settings["strategy_mc_samples"] = int(self.mc_samples_spin.value())
        settings["strategy_mc_num_opponents"] = int(self.mc_opp_spin.value())
        settings["strategy_mc_candidate_portfolios"] = int(self.mc_candidates_spin.value())
        settings["strategy_search_candidates"] = int(self.search_candidates_spin.value())
        settings["strategy_risk_a_vol"] = float(self.risk_a_spin.value())
        settings["strategy_risk_b_drawdown"] = float(self.risk_b_spin.value())
        settings["strategy_risk_c_release"] = float(self.risk_c_spin.value())
        settings["strategy_risk_max_penalty"] = float(self.risk_max_spin.value())
        settings["strategy_risk_release_window_days"] = int(self.risk_window_spin.value())
        settings["strategy_diversification_penalty"] = float(self.div_pen_spin.value())
        settings["strategy_correlation_penalty"] = float(self.corr_pen_spin.value())
        settings["strategy_bootstrap_samples"] = int(self.bootstrap_samples_spin.value())
        settings["strategy_mc_opponent_noise"] = float(self.mc_noise_spin.value())
        settings["strategy_mc_aggression_sd"] = float(self.mc_aggr_spin.value())
        settings["strategy_mc_concentration_threshold"] = float(self.mc_conc_spin.value())
        settings["strategy_mc_seed_mode"] = self.seed_mode_combo.currentText().strip().lower()
        settings["strategy_mc_random_seed"] = int(self.seed_spin.value())
        settings["strategy_corr_simulation_mode"] = self.corr_mode_combo.currentText().strip().lower()
        settings["strategy_corr_shrinkage"] = float(self.corr_shrink_spin.value())
        settings["strategy_corr_min_history_points"] = int(self.corr_min_points_spin.value())
        settings["strategy_corr_floor"] = float(self.corr_floor_spin.value())
        settings["strategy_corr_t_df"] = int(self.corr_t_df_spin.value())
        settings["strategy_search_mode"] = self.search_mode_combo.currentText().strip().lower()
        settings["strategy_search_local_iters"] = int(self.search_local_iters_spin.value())
        settings["strategy_search_population"] = int(self.search_population_spin.value())
        settings["strategy_search_generations"] = int(self.search_generations_spin.value())
        settings["strategy_search_elite_frac"] = float(self.search_elite_frac_spin.value())
        settings["strategy_search_mutation_rate"] = float(self.search_mutation_rate_spin.value())
        settings["strategy_opponent_profile"] = self.opponent_profile_combo.currentText().strip().lower()
        settings["strategy_opponent_bidup_strength"] = float(self.opp_bidup_spin.value())
        settings["strategy_opponent_cash_conservation"] = float(self.opp_cash_conserve_spin.value())
        settings["strategy_user_name"] = self.user_combo.currentText().strip()
        return settings

    def on_save_strategy_profile(self):
        name, ok_pressed = QtWidgets.QInputDialog.getText(
            self,
            "Save Strategy Profile",
            "Profile name:",
        )
        if not ok_pressed:
            return
        settings = self._collect_dashboard_strategy_settings_from_controls()
        ok, msg = storage.save_strategy_profile(str(name), settings)
        if ok:
            self._refresh_profile_combo(str(name).strip())
            self._show_info(msg)
            self._set_status(msg)
        else:
            self._show_error(msg)

    def on_load_strategy_profile(self):
        name = str(self.profile_combo.currentData() or "").strip()
        if not name:
            self._show_info("Select a saved profile first.")
            return
        payload = storage.load_strategy_profile(name)
        if not payload:
            self._show_error(f"Profile '{name}' not found or empty.")
            return
        settings = storage.load_strategy_runtime_defaults()
        settings.update(payload)
        people = storage.load_people()
        state = storage.ensure_auction_state_seeded()
        self._apply_dashboard_settings_to_controls(settings, people, state)
        self._set_status(f"Loaded strategy profile: {name}")

    def refresh_dashboard_inputs(self):
        settings = storage.load_strategy_runtime_defaults()
        people = storage.load_people()
        state = storage.ensure_auction_state_seeded()
        self._apply_dashboard_settings_to_controls(settings, people, state)
        self._refresh_profile_combo()

    def refresh_dashboard_output(self):
        self.dashboard_model.set_dataframe(pd.DataFrame())
        self.selected_model.set_dataframe(pd.DataFrame())
        self._autosize_table_columns(self.dashboard_table)
        self._autosize_table_columns(self.selected_table)
        self.dashboard_summary.setPlainText(
            "Run Dashboard to compute strategy metrics, optimizer recommendations, and Monte Carlo diagnostics."
        )

    def run_dashboard(self):
        try:
            movies_df = storage.load_movies()
            people_df = storage.load_people()
            cache_df = storage.load_cache()
            if movies_df.empty:
                self._show_error("No movies in pool. Add tickers first.")
                return
            if cache_df.empty:
                self._show_error("No HSX cache found. Scrape latest prices first.")
                return

            settings = storage.load_strategy_runtime_defaults()
            preset = self.preset_combo.currentText().strip().lower()
            preset_defaults = calculations.get_strategy_presets().get(preset, {})
            override_keys = set(settings.get("strategy_override_keys", []))

            def _set_with_override(key: str, value):
                settings[key] = value
                if key in preset_defaults:
                    base_v = preset_defaults.get(key)
                    # Numeric tolerance for float controls.
                    try:
                        is_override = abs(float(value) - float(base_v)) > 1e-9
                    except Exception:
                        is_override = value != base_v
                    if is_override:
                        override_keys.add(key)
                    elif key in override_keys:
                        override_keys.remove(key)
                else:
                    override_keys.add(key)

            risk_model = self.risk_combo.currentText().strip().lower()
            objective = self.objective_combo.currentText().strip().lower()
            integer_mode = bool(self.integer_check.isChecked())
            previous_bid = int(self.prev_bid_spin.value())
            budget_mode = self.budget_mode_combo.currentText().strip().lower()
            cost_col = self.cost_combo.currentText().strip()
            custom_budget = float(self.custom_budget_spin.value()) if budget_mode == "custom" else None
            if budget_mode == "custom":
                budget_mode = "personal"

            settings["strategy_active_preset"] = preset
            settings["strategy_risk_model"] = risk_model
            settings["strategy_objective"] = objective
            settings["strategy_integer_bid_mode"] = integer_mode
            settings["strategy_integer_prev_bid"] = previous_bid
            settings["strategy_budget_mode_preference"] = self.budget_mode_combo.currentText().strip().lower()
            settings["strategy_custom_budget_amount"] = float(self.custom_budget_spin.value())
            settings["strategy_optimizer_cost_col"] = cost_col
            _set_with_override("strategy_bid_multiplier", float(self.bid_mult_spin.value()))
            _set_with_override("strategy_max_budget_pct_per_film", float(self.base_cap_spin.value()))
            _set_with_override("strategy_market_fair_stresstest_cap", float(self.fair_cap_spin.value()))
            _set_with_override("strategy_enable_quality_filters", bool(self.quality_filter_check.isChecked()))
            _set_with_override("strategy_min_prob_positive_edge", float(self.min_edge_spin.value()))
            _set_with_override("strategy_max_prob_large_drawdown", float(self.max_dd_spin.value()))
            _set_with_override("strategy_drawdown_threshold", float(self.dd_threshold_spin.value()))
            _set_with_override("strategy_mc_samples", int(self.mc_samples_spin.value()))
            _set_with_override("strategy_mc_num_opponents", int(self.mc_opp_spin.value()))
            _set_with_override("strategy_mc_candidate_portfolios", int(self.mc_candidates_spin.value()))
            _set_with_override("strategy_risk_a_vol", float(self.risk_a_spin.value()))
            _set_with_override("strategy_risk_b_drawdown", float(self.risk_b_spin.value()))
            _set_with_override("strategy_risk_c_release", float(self.risk_c_spin.value()))
            _set_with_override("strategy_risk_max_penalty", float(self.risk_max_spin.value()))
            _set_with_override("strategy_risk_release_window_days", int(self.risk_window_spin.value()))
            _set_with_override("strategy_diversification_penalty", float(self.div_pen_spin.value()))
            _set_with_override("strategy_correlation_penalty", float(self.corr_pen_spin.value()))
            _set_with_override("strategy_bootstrap_samples", int(self.bootstrap_samples_spin.value()))
            _set_with_override("strategy_mc_opponent_noise", float(self.mc_noise_spin.value()))
            _set_with_override("strategy_mc_aggression_sd", float(self.mc_aggr_spin.value()))
            _set_with_override("strategy_mc_concentration_threshold", float(self.mc_conc_spin.value()))
            _set_with_override("strategy_mc_seed_mode", self.seed_mode_combo.currentText().strip().lower())
            _set_with_override("strategy_mc_random_seed", int(self.seed_spin.value()))
            _set_with_override("strategy_corr_simulation_mode", self.corr_mode_combo.currentText().strip().lower())
            _set_with_override("strategy_corr_shrinkage", float(self.corr_shrink_spin.value()))
            _set_with_override("strategy_corr_min_history_points", int(self.corr_min_points_spin.value()))
            _set_with_override("strategy_corr_floor", float(self.corr_floor_spin.value()))
            _set_with_override("strategy_corr_t_df", int(self.corr_t_df_spin.value()))
            _set_with_override("strategy_search_mode", self.search_mode_combo.currentText().strip().lower())
            _set_with_override("strategy_search_candidates", int(self.search_candidates_spin.value()))
            _set_with_override("strategy_search_local_iters", int(self.search_local_iters_spin.value()))
            _set_with_override("strategy_search_population", int(self.search_population_spin.value()))
            _set_with_override("strategy_search_generations", int(self.search_generations_spin.value()))
            _set_with_override("strategy_search_elite_frac", float(self.search_elite_frac_spin.value()))
            _set_with_override("strategy_search_mutation_rate", float(self.search_mutation_rate_spin.value()))
            _set_with_override("strategy_opponent_profile", self.opponent_profile_combo.currentText().strip().lower())
            _set_with_override("strategy_opponent_bidup_strength", float(self.opp_bidup_spin.value()))
            _set_with_override("strategy_opponent_cash_conservation", float(self.opp_cash_conserve_spin.value()))
            settings["strategy_override_keys"] = sorted(str(k) for k in override_keys)
            user_name = self.user_combo.currentText().strip()
            if user_name:
                settings["strategy_user_name"] = user_name
            run_seed, run_seed_mode = self._resolve_runtime_seed_for_run(settings)
            run_settings = dict(settings)
            run_settings["strategy_runtime_seed"] = int(run_seed)

            dashboard, meta, tuned = calculations.build_strategy_dashboard(
                hsx_df=cache_df,
                movies_df=movies_df,
                people_df=people_df,
                settings=run_settings,
                preset=preset,
                budget_mode=budget_mode,
                custom_budget=custom_budget,
                risk_model=risk_model,
                integer_bid_mode=integer_mode,
                previous_bid=previous_bid,
            )
            if dashboard.empty:
                self._show_error("No strategy rows available.")
                return

            budget_info = meta.get("budget_info", {})
            basis_budget = float(budget_info.get("basis_budget", 0.0))
            if basis_budget <= 0:
                self._show_error("Budget basis resolved to 0. Configure people/custom budget.")
                return

            if objective == "win_probability":
                opt = calculations.optimize_portfolio_by_win_probability(
                    strategy_df=dashboard,
                    budget=basis_budget,
                    settings=tuned,
                    cost_col=cost_col,
                    risk_model=risk_model,
                )
            else:
                opt = calculations.optimize_portfolio(
                    strategy_df=dashboard,
                    budget=basis_budget,
                    settings=tuned,
                    cost_col=cost_col,
                )

            if opt.get("num_selected", 0) == 0 and cost_col == "market_fair_bid":
                if objective == "win_probability":
                    opt = calculations.optimize_portfolio_by_win_probability(
                        strategy_df=dashboard,
                        budget=basis_budget,
                        settings=tuned,
                        cost_col="target_bid",
                        risk_model=risk_model,
                    )
                else:
                    opt = calculations.optimize_portfolio(
                        strategy_df=dashboard,
                        budget=basis_budget,
                        settings=tuned,
                        cost_col="target_bid",
                    )
            win_eval = None
            if objective != "win_probability":
                try:
                    win_eval = calculations.estimate_portfolio_win_probability(
                        strategy_df=dashboard,
                        portfolio_df=opt.get("selected", pd.DataFrame()),
                        budget=basis_budget,
                        settings=tuned,
                        risk_model=risk_model,
                        cost_col=opt.get("cost_col", cost_col),
                    )
                except Exception:
                    win_eval = None

            sim = calculations.simulate_portfolio_monte_carlo(
                strategy_df=dashboard,
                portfolio_df=opt.get("selected", pd.DataFrame()),
                settings=tuned,
                risk_model=risk_model,
            )

            selected_tickers = set(opt.get("selected", pd.DataFrame()).get("ticker", pd.Series(dtype=object)).astype(str))
            dashboard = dashboard.copy()
            dashboard["optimizer_selected"] = dashboard["ticker"].astype(str).isin(selected_tickers)
            avail = dashboard[dashboard["owner"].fillna("") == ""].copy()
            avail = avail.sort_values(
                ["priority_score", "prob_positive_edge", "adjusted_expected"],
                ascending=[False, False, False],
                na_position="last",
            )

            display_cols = [
                "ticker", "current_price", "adjusted_expected",
                "target_bid_int" if integer_mode else "target_bid",
                "target_market_bid_int" if integer_mode else "target_market_bid",
                "max_bid_int" if integer_mode else "max_bid",
                "risk_penalty", "prob_positive_edge", "prob_large_drawdown",
                "market_value_ratio", "priority_score", "optimizer_selected",
            ]
            display_df = avail[[c for c in display_cols if c in avail.columns]].copy()
            display_df = display_df.rename(columns={
                "target_market_bid_int": "target_mkt_bid_int",
                "target_market_bid": "target_mkt_bid",
            })
            self.dashboard_model.set_dataframe(display_df)
            self._autosize_table_columns(self.dashboard_table)

            selected = opt.get("selected", pd.DataFrame()).copy()
            sel_cols = [c for c in ["ticker", "target_bid", "optimizer_cost", "adjusted_expected", "eff_per_dollar"] if c in selected.columns]
            self.selected_model.set_dataframe(selected[sel_cols] if sel_cols else selected)
            self._autosize_table_columns(self.selected_table)

            def _pct(value) -> str:
                v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                if pd.isna(v):
                    return "N/A"
                return f"{100.0 * float(v):.1f}%"

            def _num(value, decimals: int = 2, default: str = "N/A") -> str:
                v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                if pd.isna(v):
                    return default
                return f"{float(v):.{decimals}f}"

            selected_df = opt.get("selected", pd.DataFrame()).copy()
            alternates_df = opt.get("alternates", pd.DataFrame()).copy()
            if not selected_df.empty and "eff_per_dollar" not in selected_df.columns:
                c = pd.to_numeric(selected_df.get("optimizer_cost", np.nan), errors="coerce")
                a = pd.to_numeric(selected_df.get("adjusted_expected", np.nan), errors="coerce")
                selected_df["eff_per_dollar"] = np.where(c > 0, a / c, np.nan)
            if not alternates_df.empty and "eff_per_dollar" not in alternates_df.columns:
                c = pd.to_numeric(alternates_df.get("optimizer_cost", np.nan), errors="coerce")
                a = pd.to_numeric(alternates_df.get("adjusted_expected", np.nan), errors="coerce")
                alternates_df["eff_per_dollar"] = np.where(c > 0, a / c, np.nan)
            eval_meta = opt if objective == "win_probability" else (win_eval if isinstance(win_eval, dict) else {})

            text = [
                "=== Draft Strategy Dashboard ===",
                f"Preset: {meta.get('preset', 'balanced')} | Budget basis: {_fmt_money(basis_budget)} ({budget_info.get('budget_mode', 'personal')})",
                f"Risk model: {risk_model} | Objective: {objective} | Integer bids: {'on' if integer_mode else 'off'}",
                f"Seed mode: {run_seed_mode} | Seed used: {run_seed}",
                f"Cost basis: {cost_col}",
            ]
            if integer_mode:
                text.append(f"Integer bid rules: whole number, > {previous_bid}, <= remaining budget")
            mpf = pd.to_numeric(pd.Series([budget_info.get("market_pressure_factor", np.nan)]), errors="coerce").iloc[0]
            if pd.notna(mpf):
                text.append(f"Market pressure factor (league/personal): {float(mpf):.2f}")

            text.extend([
                "",
                "Optimizer summary:",
                f"Selected: {opt.get('num_selected', 0)} movie(s) | Spend: {_fmt_money(opt.get('total_spend', 0.0))} / {_fmt_money(opt.get('budget', basis_budget))} | Left: {_fmt_money(opt.get('leftover', 0.0))}",
                f"AdjExp total: {opt.get('total_adjusted_expected', 0.0):.2f}",
                f"Per-film cap: {100.0 * float(opt.get('max_budget_pct_per_film', np.nan)):.1f}% of budget",
            ])
            if objective == "win_probability":
                text.append(
                    f"Estimated win probability: {_pct(opt.get('win_probability'))} "
                    f"(vs {int(opt.get('opponent_count', 0))} simulated opponents)"
                )
            elif isinstance(win_eval, dict):
                text.append(
                    f"Estimated win probability (selected portfolio): {_pct(win_eval.get('win_probability'))} "
                    f"(vs {int(win_eval.get('opponent_count', 0))} simulated opponents)"
                )
            qf = opt.get("quality_filter_info", {})
            if isinstance(qf, dict) and qf.get("enabled"):
                text.append(
                    f"Quality filter: P(Edge)>={100.0 * float(qf.get('min_prob_positive_edge', 0.0)):.0f}% "
                    f"and P(DD)<={100.0 * float(qf.get('max_prob_large_drawdown', 1.0)):.0f}% "
                    f"(kept {qf.get('rows_after', 0)}/{qf.get('rows_before', 0)})"
                )
            else:
                text.append("Quality filter: off")
            if integer_mode:
                drift = float(pd.to_numeric(dashboard.get("integer_rounding_drift", np.nan), errors="coerce").sum())
                no_legal = int((pd.to_numeric(dashboard.get("can_bid_int", True), errors="coerce") == 0).sum())
                text.append(f"Integer mode: total rounding drift={drift:+.2f}  no-legal-bid rows={no_legal}")

            text.extend(["", "Selected portfolio (top picks):"])
            if selected_df.empty:
                text.append("None selected.")
            else:
                for _, row in selected_df.head(8).iterrows():
                    text.append(
                        f"{str(row.get('ticker', '')):<8}  "
                        f"bid={_num(row.get('target_bid'))}  "
                        f"cost={_num(row.get('optimizer_cost'))}  "
                        f"adj={_num(row.get('adjusted_expected'))}  "
                        f"eff/$={_num(row.get('eff_per_dollar'))}"
                    )

            text.extend(["", "Top alternates:"])
            if alternates_df.empty:
                text.append("No alternates available.")
            else:
                for _, row in alternates_df.head(5).iterrows():
                    text.append(
                        f"{str(row.get('ticker', '')):<8}  "
                        f"bid={_num(row.get('target_bid'))}  "
                        f"cost={_num(row.get('optimizer_cost'))}  "
                        f"adj={_num(row.get('adjusted_expected'))}  "
                        f"eff/$={_num(row.get('eff_per_dollar'))}"
                    )

            text.extend([
                "",
                "Model settings:",
                f"Risk penalty coeffs: A={float(tuned.get('strategy_risk_a_vol', np.nan)):.2f}  "
                f"B={float(tuned.get('strategy_risk_b_drawdown', np.nan)):.2f}  "
                f"C={float(tuned.get('strategy_risk_c_release', np.nan)):.2f}  "
                f"max={100.0 * float(tuned.get('strategy_risk_max_penalty', np.nan)):.1f}%  "
                f"window={int(tuned.get('strategy_risk_release_window_days', 30))}d",
                f"Portfolio penalties: div={float(tuned.get('strategy_diversification_penalty', 0.0)):.2f}  "
                f"corr={float(tuned.get('strategy_correlation_penalty', 0.0)):.2f}",
                f"Search mode={str(eval_meta.get('search_mode_used', tuned.get('strategy_search_mode', 'current_sampled')))}  "
                f"candidates={int(pd.to_numeric(pd.Series([eval_meta.get('candidate_count_evaluated', 0)]), errors='coerce').fillna(0).iloc[0])}  "
                f"runtime_ms={float(pd.to_numeric(pd.Series([eval_meta.get('search_runtime_ms', np.nan)]), errors='coerce').fillna(0.0).iloc[0]):.1f}",
                f"Opponent profile={str(eval_meta.get('opponent_profile_used', tuned.get('strategy_opponent_profile', 'balanced_field')))}  "
                f"bidup={float(tuned.get('strategy_opponent_bidup_strength', 0.0)):.2f}  "
                f"cash_conserve={float(tuned.get('strategy_opponent_cash_conservation', 0.0)):.2f}",
                "",
                "Monte Carlo portfolio simulation:",
                f"samples={sim.get('samples', 0)}  gross mean/p10/p50/p90="
                f"{sim.get('gross_mean', 0.0):.2f} / {sim.get('gross_p10', 0.0):.2f} / "
                f"{sim.get('gross_p50', 0.0):.2f} / {sim.get('gross_p90', 0.0):.2f}",
                f"mc_seed={sim.get('seed_mode', run_seed_mode)}:{int(sim.get('seed_used', run_seed))}",
                f"corr_mode requested/used={sim.get('corr_mode_requested', 'independent')}/{sim.get('corr_mode_used', 'independent')}  "
                f"dim={int(sim.get('corr_effective_dim', 0))}  "
                f"fallback={str(sim.get('corr_fallback_reason', '') or '-')}",
                f"prob_samples={int(tuned.get('strategy_bootstrap_samples', 1000))}  "
                f"opp_noise={float(tuned.get('strategy_mc_opponent_noise', 0.30)):.2f}  "
                f"opp_aggr_sd={float(tuned.get('strategy_mc_aggression_sd', 0.10)):.2f}  "
                f"conc_threshold={100.0 * float(tuned.get('strategy_mc_concentration_threshold', 0.40)):.1f}%",
                f"exchange_rate={sim.get('exchange_rate_million_per_auction_dollar', np.nan):.4f} gross-units per $1 auction",
                f"budget-equiv gross mean/p10/p50/p90="
                f"{sim.get('gross_budget_equiv_mean', np.nan):.2f} / {sim.get('gross_budget_equiv_p10', np.nan):.2f} / "
                f"{sim.get('gross_budget_equiv_p50', np.nan):.2f} / {sim.get('gross_budget_equiv_p90', np.nan):.2f}",
                f"P(gross_equiv < spend)={_pct(sim.get('prob_gross_below_spend'))}  "
                f"Concentration downside={_pct(sim.get('concentration_downside_prob'))}",
            ])

            # Practical recommendations from current run metrics.
            recs = []
            p_below = pd.to_numeric(pd.Series([sim.get("prob_gross_below_spend", np.nan)]), errors="coerce").iloc[0]
            p_conc = pd.to_numeric(pd.Series([sim.get("concentration_downside_prob", np.nan)]), errors="coerce").iloc[0]
            left = pd.to_numeric(pd.Series([opt.get("leftover", np.nan)]), errors="coerce").iloc[0]
            win_p = np.nan
            if objective == "win_probability":
                win_p = pd.to_numeric(pd.Series([opt.get("win_probability", np.nan)]), errors="coerce").iloc[0]
            elif isinstance(win_eval, dict):
                win_p = pd.to_numeric(pd.Series([win_eval.get("win_probability", np.nan)]), errors="coerce").iloc[0]

            if pd.notna(p_below):
                if p_below > 0.20:
                    recs.append("Portfolio downside is elevated. Tighten caps or increase quality filter strictness.")
                elif p_below < 0.05:
                    recs.append("Downside risk is controlled. You can consider modestly higher aggression if needed.")
            if pd.notna(p_conc) and p_conc > 0.15:
                recs.append("Concentration downside is high. Lower per-film cap or raise diversification/correlation penalties.")
            if pd.notna(win_p):
                if win_p < 0.20:
                    recs.append("Win probability is low. For stress tests, consider higher market-fair cap (0.50-0.60) and fewer low-edge fillers.")
                elif win_p > 0.40:
                    recs.append("Win probability is competitive. Prioritize execution discipline at/under target bids.")
            if pd.notna(left) and left > 0.15 * basis_budget:
                recs.append("Large leftover budget detected. Check cap/filter settings; you may be leaving too much value unallocated.")
            if isinstance(qf, dict) and not qf.get("enabled", False):
                recs.append("Quality filter is off. Enabling it can reduce low-edge/high-drawdown filler picks.")
            if not recs:
                recs.append("Current setup is balanced. Validate robustness by running a few seeds and comparing top selections.")

            text.extend(["", "Practical recommendations:"])
            for rec in recs[:5]:
                text.append(f"- {rec}")

            diag = meta.get("diagnostics", {})
            text.extend([
                "",
                "Diagnostics:",
                f"rows={int(diag.get('rows', 0))}  missing_price={int(diag.get('missing_current_price', 0))}  "
                f"missing_history={int(diag.get('missing_history', 0))}  missing_probs={int(diag.get('missing_probs', 0))}",
            ])
            validation = meta.get("validation", {})
            stability = validation.get("ranking_stability", {})
            if isinstance(stability, dict) and stability:
                text.append(
                    f"ranking_stability(top{int(stability.get('top_n', 10))} overlap mean/min/max): "
                    f"{_pct(stability.get('mean_top_overlap'))} / {_pct(stability.get('min_top_overlap'))} / {_pct(stability.get('max_top_overlap'))}"
                )
            forward = validation.get("forward_check", {})
            if isinstance(forward, dict) and forward:
                ic = pd.to_numeric(pd.Series([forward.get("information_coefficient", np.nan)]), errors="coerce").iloc[0]
                ic_txt = "N/A" if pd.isna(ic) else f"{float(ic):.3f}"
                text.append(
                    f"forward_check(h={int(forward.get('horizon_days', 0))}d, samples={int(forward.get('samples', 0))}): "
                    f"dir_acc={_pct(forward.get('directional_accuracy'))}  IC={ic_txt}"
                )

            text.extend([
                "",
                "Glossary:",
                "- adjusted_expected: risk-adjusted expectation from current price and history features",
                "- target_bid / target_bid_int: recommended bid after risk penalty and budget share",
                "- target_market_bid / target_market_bid_int: market-pressure scaled target (same risk logic, league-liquidity scale)",
                "- prob_positive_edge: probability adjusted_expected exceeds bid threshold",
                "- prob_large_drawdown: probability of large downside outcome from simulations",
                "- market_value_ratio: adjusted_expected/current_price",
                "- priority_score: blended ranking score used for selection",
            ])
            if cost_col == "current_price":
                text.insert(4, "NOTE: current_price is diagnostic-only and not auction-dollar scaled.")
            self.dashboard_summary.setPlainText("\n".join(text))
            self._set_status("Dashboard updated.")
        except Exception as exc:
            self._show_error("Failed to run dashboard.", detail=f"{exc}\n\n{traceback.format_exc()}")

    # ---------- Draft State ----------

    def _remaining_movies_sorted(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        if movies_df.empty:
            return pd.DataFrame(columns=["ticker", "release_date"])
        out = movies_df[movies_df["owner"].fillna("") == ""][["ticker", "release_date"]].copy()
        out["ticker"] = out["ticker"].astype(str).str.upper()
        out["_release_sort"] = pd.to_datetime(out["release_date"], errors="coerce")
        out = out.sort_values("_release_sort", ascending=True, na_position="last")
        return out.drop(columns=["_release_sort"])

    def _build_draft_state_tab(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Draft State")
        root = QtWidgets.QVBoxLayout(tab)

        top = QtWidgets.QHBoxLayout()
        root.addLayout(top)
        top.addWidget(QtWidgets.QLabel("Current Movie"))
        self.state_movie_combo = QtWidgets.QComboBox()
        top.addWidget(self.state_movie_combo)
        top.addWidget(QtWidgets.QLabel("Current Bid"))
        self.state_bid_spin = QtWidgets.QSpinBox()
        self.state_bid_spin.setRange(0, 100000)
        top.addWidget(self.state_bid_spin)
        btn_save = QtWidgets.QPushButton("Save Context")
        btn_save.clicked.connect(self.on_save_context)
        top.addWidget(btn_save)
        btn_clear = QtWidgets.QPushButton("Clear Context")
        btn_clear.clicked.connect(self.on_clear_context)
        top.addWidget(btn_clear)

        ops = QtWidgets.QGridLayout()
        root.addLayout(ops)
        self.assign_ticker_combo = QtWidgets.QComboBox()
        self.assign_user_combo = QtWidgets.QComboBox()
        self.assign_price_spin = QtWidgets.QDoubleSpinBox()
        self.assign_price_spin.setRange(0.0, 50000.0)
        self.assign_price_spin.setDecimals(2)
        self.assign_price_spin.setSingleStep(1.0)
        self.unassign_ticker_combo = QtWidgets.QComboBox()

        ops.addWidget(QtWidgets.QLabel("Assign Ticker"), 0, 0)
        ops.addWidget(self.assign_ticker_combo, 0, 1)
        ops.addWidget(QtWidgets.QLabel("Winner"), 0, 2)
        ops.addWidget(self.assign_user_combo, 0, 3)
        ops.addWidget(QtWidgets.QLabel("Final Price (0=skip)"), 0, 4)
        ops.addWidget(self.assign_price_spin, 0, 5)
        btn_assign = QtWidgets.QPushButton("Assign")
        btn_assign.clicked.connect(self.on_assign_movie)
        ops.addWidget(btn_assign, 0, 6)

        ops.addWidget(QtWidgets.QLabel("Unassign Ticker"), 1, 0)
        ops.addWidget(self.unassign_ticker_combo, 1, 1)
        btn_unassign = QtWidgets.QPushButton("Unassign")
        btn_unassign.clicked.connect(self.on_unassign_movie)
        ops.addWidget(btn_unassign, 1, 2)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(splitter, 1)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.addWidget(QtWidgets.QLabel("Assigned Movies"))
        self.assigned_table = QtWidgets.QTableView()
        self.assigned_model = DataFrameTableModel(pd.DataFrame())
        self.assigned_table.setModel(self.assigned_model)
        self.assigned_table.horizontalHeader().setStretchLastSection(True)
        left_layout.addWidget(self.assigned_table)
        splitter.addWidget(left)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.addWidget(QtWidgets.QLabel("Assignment History"))
        self.assign_hist_table = QtWidgets.QTableView()
        self.assign_hist_model = DataFrameTableModel(pd.DataFrame())
        self.assign_hist_table.setModel(self.assign_hist_model)
        self.assign_hist_table.horizontalHeader().setStretchLastSection(True)
        right_layout.addWidget(self.assign_hist_table)
        splitter.addWidget(right)
        splitter.setSizes([720, 720])

    def refresh_draft_state(self):
        state = storage.ensure_auction_state_seeded()
        cur_movie = str(state.get("current_movie", "") or "").strip().upper()
        self.state_bid_spin.setValue(int(max(state.get("current_bid", 0), 0)))

        movies = storage.load_movies()
        people = storage.load_people()
        cache = storage.load_cache()
        assigned = storage.get_assigned_movies_df()
        hist = storage.get_assignment_history_df()

        if not assigned.empty and not cache.empty:
            assigned = assigned.merge(cache[["ticker", "name"]], on="ticker", how="left")
            assigned["movie"] = assigned.apply(lambda r: _clean_name(r.get("name"), r.get("ticker")), axis=1)

        assigned_cols = [c for c in ["ticker", "movie", "winner", "final_price", "assigned_at", "budget_applied"] if c in assigned.columns]
        self.assigned_model.set_dataframe(assigned[assigned_cols] if assigned_cols else assigned)

        hist_cols = [c for c in ["timestamp", "event_type", "ticker", "winner", "final_price", "source", "notes"] if c in hist.columns]
        self.assign_hist_model.set_dataframe(hist[hist_cols] if hist_cols else hist)

        unassigned = self._remaining_movies_sorted(movies)

        self.state_movie_combo.blockSignals(True)
        self.state_movie_combo.clear()
        self.state_movie_combo.addItem("(none)", "")
        for _, row in unassigned.iterrows():
            ticker = str(row.get("ticker", "")).upper()
            rel = str(row.get("release_date", "") or "").strip()
            label = f"{ticker}  |  {rel}" if rel else f"{ticker}  |  (no release date)"
            self.state_movie_combo.addItem(label, ticker)
        if cur_movie:
            idx = self.state_movie_combo.findData(cur_movie)
            if idx >= 0:
                self.state_movie_combo.setCurrentIndex(idx)
            else:
                self.state_movie_combo.addItem(f"{cur_movie}  |  (not in remaining pool)", cur_movie)
                self.state_movie_combo.setCurrentIndex(self.state_movie_combo.count() - 1)
        else:
            self.state_movie_combo.setCurrentIndex(0)
        self.state_movie_combo.blockSignals(False)

        self.assign_ticker_combo.blockSignals(True)
        self.assign_ticker_combo.clear()
        for _, row in unassigned.iterrows():
            ticker = str(row.get("ticker", "")).upper()
            rel = str(row.get("release_date", "") or "").strip()
            label = f"{ticker}  |  {rel}" if rel else f"{ticker}  |  (no release date)"
            self.assign_ticker_combo.addItem(label, ticker)
        self.assign_ticker_combo.blockSignals(False)

        self.assign_user_combo.blockSignals(True)
        self.assign_user_combo.clear()
        self.assign_user_combo.addItems(people.get("name", pd.Series(dtype=object)).astype(str).tolist())
        self.assign_user_combo.blockSignals(False)

        self.unassign_ticker_combo.blockSignals(True)
        self.unassign_ticker_combo.clear()
        self.unassign_ticker_combo.addItems(assigned.get("ticker", pd.Series(dtype=object)).astype(str).str.upper().tolist())
        self.unassign_ticker_combo.blockSignals(False)

    def on_save_context(self):
        ticker = str(self.state_movie_combo.currentData() or "").strip().upper()
        current_bid = int(self.state_bid_spin.value())
        storage.set_current_auction_context(ticker=ticker, current_bid=current_bid)
        self._set_status(f"Saved draft context: movie={ticker or '-'} bid={current_bid}")
        self.refresh_all()

    def on_clear_context(self):
        storage.set_current_auction_context("", 0)
        self._set_status("Cleared draft context.")
        self.refresh_all()

    def on_assign_movie(self):
        ticker = str(self.assign_ticker_combo.currentData() or "").strip().upper()
        winner = self.assign_user_combo.currentText().strip()
        final_price = float(self.assign_price_spin.value())
        final_price = final_price if final_price > 0 else None
        if not ticker or not winner:
            self._show_error("Ticker and winner are required.")
            return
        ok, msg, _ = storage.assign_movie(
            ticker=ticker,
            winner=winner,
            final_price=final_price,
            source="gui_assign",
        )
        if ok:
            self._set_status(msg)
            self.refresh_all()
        else:
            self._show_error(msg)

    def on_unassign_movie(self):
        ticker = self.unassign_ticker_combo.currentText().strip().upper()
        if not ticker:
            self._show_error("Choose an assigned ticker.")
            return
        ok, msg, _ = storage.unassign_movie(
            ticker=ticker,
            source="gui_unassign",
            restore_budget=True,
        )
        if ok:
            self._set_status(msg)
            self.refresh_all()
        else:
            self._show_error(msg)

    # ---------- History ----------

    def _build_history_tab(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "History")
        root = QtWidgets.QVBoxLayout(tab)
        top = QtWidgets.QHBoxLayout()
        root.addLayout(top)
        top.addWidget(QtWidgets.QLabel("Ticker"))
        self.hist_ticker_combo = QtWidgets.QComboBox()
        top.addWidget(self.hist_ticker_combo)
        btn_plot = QtWidgets.QPushButton("Plot")
        btn_plot.clicked.connect(self.plot_history)
        top.addWidget(btn_plot)
        btn_refresh = QtWidgets.QPushButton("Refresh")
        btn_refresh.clicked.connect(self.refresh_history)
        top.addWidget(btn_refresh)
        top.addStretch(1)

        self.hist_canvas = MplCanvas()
        root.addWidget(self.hist_canvas, 1)

    def refresh_history(self):
        movies = storage.load_movies()
        tickers = []
        for t in movies.get("ticker", pd.Series(dtype=object)).astype(str).str.upper().unique():
            df = storage.load_price_history(t)
            if df is not None and not df.empty and "date" in df.columns and "price" in df.columns:
                tickers.append(t)
        tickers = sorted(set(tickers))
        self.hist_ticker_combo.blockSignals(True)
        self.hist_ticker_combo.clear()
        self.hist_ticker_combo.addItems(tickers)
        self.hist_ticker_combo.blockSignals(False)
        if tickers:
            self.plot_history()
        else:
            self.hist_canvas.ax.clear()
            self.hist_canvas.ax.set_title("No history data available")
            self.hist_canvas.draw_idle()

    def plot_history(self):
        ticker = self.hist_ticker_combo.currentText().strip().upper()
        if not ticker:
            return
        df = storage.load_price_history(ticker)
        if df is None or df.empty:
            return
        data = df.copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data["price"] = pd.to_numeric(data["price"], errors="coerce")
        data = data.dropna(subset=["date", "price"]).sort_values("date")
        if data.empty:
            return
        cache = storage.load_cache()
        name = ticker
        if not cache.empty:
            row = cache[cache["ticker"].astype(str).str.upper() == ticker]
            if not row.empty:
                name = _clean_name(row.iloc[0].get("name"), ticker)
        self.hist_canvas.ax.clear()
        self.hist_canvas.ax.plot(data["date"], data["price"], linewidth=2.0)
        self.hist_canvas.ax.set_title(f"{ticker} - {name}")
        self.hist_canvas.ax.set_xlabel("Date")
        self.hist_canvas.ax.set_ylabel("Price")
        self.hist_canvas.ax.grid(alpha=0.25)
        self.hist_canvas.figure.autofmt_xdate()
        self.hist_canvas.draw_idle()

    # ---------- Data ----------

    def _build_data_tab(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Data")
        root = QtWidgets.QVBoxLayout(tab)

        actions = QtWidgets.QHBoxLayout()
        root.addLayout(actions)
        btn_scrape = QtWidgets.QPushButton("Scrape Latest Prices")
        btn_scrape.clicked.connect(self.on_scrape_prices)
        actions.addWidget(btn_scrape)
        btn_hist = QtWidgets.QPushButton("Fetch Histories For Pool")
        btn_hist.clicked.connect(self.on_fetch_histories)
        actions.addWidget(btn_hist)
        btn_dates = QtWidgets.QPushButton("Fetch Release Dates")
        btn_dates.clicked.connect(self.on_fetch_release_dates)
        actions.addWidget(btn_dates)
        actions.addStretch(1)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(splitter, 1)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.addWidget(QtWidgets.QLabel("Movies"))
        self.movies_table = QtWidgets.QTableView()
        self.movies_model = DataFrameTableModel(pd.DataFrame())
        self.movies_table.setModel(self.movies_model)
        self.movies_table.horizontalHeader().setStretchLastSection(True)
        left_layout.addWidget(self.movies_table)

        movie_ops = QtWidgets.QGridLayout()
        left_layout.addLayout(movie_ops)
        self.movie_add_edit = QtWidgets.QLineEdit()
        self.movie_add_edit.setPlaceholderText("Ticker")
        movie_ops.addWidget(QtWidgets.QLabel("Add"), 0, 0)
        movie_ops.addWidget(self.movie_add_edit, 0, 1)
        btn_add = QtWidgets.QPushButton("Add")
        btn_add.clicked.connect(self.on_add_movie)
        movie_ops.addWidget(btn_add, 0, 2)

        self.movie_bulk_edit = QtWidgets.QLineEdit()
        self.movie_bulk_edit.setPlaceholderText("A, B, C")
        movie_ops.addWidget(QtWidgets.QLabel("Bulk"), 1, 0)
        movie_ops.addWidget(self.movie_bulk_edit, 1, 1)
        btn_bulk = QtWidgets.QPushButton("Add Bulk")
        btn_bulk.clicked.connect(self.on_bulk_add_movies)
        movie_ops.addWidget(btn_bulk, 1, 2)

        self.movie_remove_edit = QtWidgets.QLineEdit()
        self.movie_remove_edit.setPlaceholderText("Ticker")
        movie_ops.addWidget(QtWidgets.QLabel("Remove"), 2, 0)
        movie_ops.addWidget(self.movie_remove_edit, 2, 1)
        btn_remove = QtWidgets.QPushButton("Remove")
        btn_remove.clicked.connect(self.on_remove_movie)
        movie_ops.addWidget(btn_remove, 2, 2)

        self.movie_rel_ticker = QtWidgets.QLineEdit()
        self.movie_rel_ticker.setPlaceholderText("Ticker")
        self.movie_rel_date = QtWidgets.QDateEdit()
        self.movie_rel_date.setDisplayFormat("yyyy-MM-dd")
        self.movie_rel_date.setCalendarPopup(True)
        self.movie_rel_date.setDate(QtCore.QDate.currentDate())
        movie_ops.addWidget(QtWidgets.QLabel("Release"), 3, 0)
        movie_ops.addWidget(self.movie_rel_ticker, 3, 1)
        movie_ops.addWidget(self.movie_rel_date, 3, 2)
        btn_set_rel = QtWidgets.QPushButton("Set Date")
        btn_set_rel.clicked.connect(self.on_set_release_date)
        movie_ops.addWidget(btn_set_rel, 3, 3)

        splitter.addWidget(left)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.addWidget(QtWidgets.QLabel("People"))
        self.people_table = QtWidgets.QTableView()
        self.people_model = DataFrameTableModel(pd.DataFrame())
        self.people_table.setModel(self.people_model)
        self.people_table.horizontalHeader().setStretchLastSection(True)
        right_layout.addWidget(self.people_table)

        people_ops = QtWidgets.QGridLayout()
        right_layout.addLayout(people_ops)
        self.person_name_edit = QtWidgets.QLineEdit()
        self.person_start_spin = QtWidgets.QDoubleSpinBox()
        self.person_start_spin.setRange(0.0, 100000.0)
        self.person_start_spin.setDecimals(2)
        self.person_start_spin.setValue(200.0)
        self.person_remain_spin = QtWidgets.QDoubleSpinBox()
        self.person_remain_spin.setRange(0.0, 100000.0)
        self.person_remain_spin.setDecimals(2)
        self.person_remain_spin.setValue(200.0)
        people_ops.addWidget(QtWidgets.QLabel("Name"), 0, 0)
        people_ops.addWidget(self.person_name_edit, 0, 1)
        people_ops.addWidget(QtWidgets.QLabel("Start"), 0, 2)
        people_ops.addWidget(self.person_start_spin, 0, 3)
        people_ops.addWidget(QtWidgets.QLabel("Remain"), 0, 4)
        people_ops.addWidget(self.person_remain_spin, 0, 5)
        btn_add_person = QtWidgets.QPushButton("Add Person")
        btn_add_person.clicked.connect(self.on_add_person)
        people_ops.addWidget(btn_add_person, 0, 6)

        self.person_select_combo = QtWidgets.QComboBox()
        self.person_update_spin = QtWidgets.QDoubleSpinBox()
        self.person_update_spin.setRange(0.0, 100000.0)
        self.person_update_spin.setDecimals(2)
        people_ops.addWidget(QtWidgets.QLabel("Person"), 1, 0)
        people_ops.addWidget(self.person_select_combo, 1, 1)
        people_ops.addWidget(QtWidgets.QLabel("New Remaining"), 1, 2)
        people_ops.addWidget(self.person_update_spin, 1, 3)
        btn_update = QtWidgets.QPushButton("Update")
        btn_update.clicked.connect(self.on_update_remaining)
        people_ops.addWidget(btn_update, 1, 4)
        btn_remove_person = QtWidgets.QPushButton("Remove Person")
        btn_remove_person.clicked.connect(self.on_remove_person)
        people_ops.addWidget(btn_remove_person, 1, 5)

        splitter.addWidget(right)
        splitter.setSizes([760, 760])

    def refresh_data_tab(self):
        movies = storage.load_movies()
        cache = storage.load_cache()
        people = storage.load_people()

        movie_display = movies.copy()
        if not movie_display.empty and not cache.empty:
            movie_display = movie_display.merge(cache[["ticker", "name", "current_price"]], on="ticker", how="left")
            movie_display["movie"] = movie_display.apply(lambda r: _clean_name(r.get("name"), r.get("ticker")), axis=1)
        movie_cols = [c for c in ["ticker", "movie", "owner", "release_date", "current_price"] if c in movie_display.columns]
        self.movies_model.set_dataframe(movie_display[movie_cols] if movie_cols else movie_display)
        self.people_model.set_dataframe(people)

        self.person_select_combo.blockSignals(True)
        self.person_select_combo.clear()
        self.person_select_combo.addItems(people.get("name", pd.Series(dtype=object)).astype(str).tolist())
        self.person_select_combo.blockSignals(False)

    def on_scrape_prices(self):
        try:
            df = scrape_hsx()
            storage.save_cache(df)
            self._set_status(f"HSX cache updated: {len(df)} rows.")
            self.refresh_all()
        except Exception as exc:
            self._show_error("Failed to scrape prices.", detail=str(exc))

    def on_fetch_histories(self):
        movies = storage.load_movies()
        if movies.empty:
            self._show_error("No movies in pool.")
            return
        tickers = movies["ticker"].dropna().astype(str).str.upper().tolist()
        ok_count = 0
        for ticker in tickers:
            df, _ = scrape_price_history(ticker)
            if df is not None:
                storage.save_price_history(ticker, df)
                ok_count += 1
            QtWidgets.QApplication.processEvents()
        self._set_status(f"Fetched histories: {ok_count}/{len(tickers)}.")
        self.refresh_all()

    def on_fetch_release_dates(self):
        movies = storage.load_movies()
        if movies.empty:
            self._show_error("No movies in pool.")
            return
        tickers = movies["ticker"].dropna().astype(str).str.upper().tolist()
        try:
            dates = scrape_release_dates(tickers)
            if dates:
                out = movies.copy()
                for ticker, date_str in dates.items():
                    out.loc[out["ticker"].astype(str).str.upper() == str(ticker).upper(), "release_date"] = date_str
                storage.save_movies(out)
                self._set_status(f"Updated release dates: {len(dates)}")
                self.refresh_all()
            else:
                self._show_info("No release dates found.")
        except Exception as exc:
            self._show_error("Failed to fetch release dates.", detail=str(exc))

    def on_add_movie(self):
        ticker = self.movie_add_edit.text().strip().upper()
        if not ticker:
            return
        movies = storage.load_movies()
        existing = movies.get("ticker", pd.Series(dtype=object)).astype(str).str.upper().values
        if ticker in existing:
            self._show_error(f"{ticker} already in pool.")
            return
        new_row = pd.DataFrame([{"ticker": ticker, "owner": "", "release_date": ""}])
        storage.save_movies(pd.concat([movies, new_row], ignore_index=True))
        self.movie_add_edit.clear()
        self.refresh_all()

    def on_bulk_add_movies(self):
        raw = self.movie_bulk_edit.text().strip()
        if not raw:
            return
        incoming = [t.strip().upper() for t in raw.split(",") if t.strip()]
        movies = storage.load_movies()
        existing = set(movies.get("ticker", pd.Series(dtype=object)).astype(str).str.upper())
        new_tickers = [t for t in incoming if t not in existing]
        if not new_tickers:
            self._show_info("No new tickers to add.")
            return
        new_rows = pd.DataFrame([{"ticker": t, "owner": "", "release_date": ""} for t in new_tickers])
        storage.save_movies(pd.concat([movies, new_rows], ignore_index=True))
        self.movie_bulk_edit.clear()
        self.refresh_all()

    def on_remove_movie(self):
        ticker = self.movie_remove_edit.text().strip().upper()
        if not ticker:
            return
        movies = storage.load_movies()
        out = movies[movies["ticker"].astype(str).str.upper() != ticker].copy()
        storage.save_movies(out)
        self.movie_remove_edit.clear()
        self.refresh_all()

    def on_set_release_date(self):
        ticker = self.movie_rel_ticker.text().strip().upper()
        if not ticker:
            return
        movies = storage.load_movies()
        if ticker not in movies.get("ticker", pd.Series(dtype=object)).astype(str).str.upper().values:
            self._show_error(f"{ticker} not in pool.")
            return
        date_str = self.movie_rel_date.date().toString("yyyy-MM-dd")
        out = movies.copy()
        out.loc[out["ticker"].astype(str).str.upper() == ticker, "release_date"] = date_str
        storage.save_movies(out)
        self.refresh_all()

    def on_add_person(self):
        name = self.person_name_edit.text().strip()
        if not name:
            return
        start = float(self.person_start_spin.value())
        remain = float(self.person_remain_spin.value())
        people = storage.load_people()
        if name in people.get("name", pd.Series(dtype=object)).astype(str).values:
            self._show_error(f"{name} already exists.")
            return
        new_row = pd.DataFrame([{"name": name, "starting_money": start, "remaining_money": remain}])
        storage.save_people(pd.concat([people, new_row], ignore_index=True))
        self.person_name_edit.clear()
        self.refresh_all()

    def on_update_remaining(self):
        name = self.person_select_combo.currentText().strip()
        if not name:
            return
        value = float(self.person_update_spin.value())
        people = storage.load_people()
        if name not in people.get("name", pd.Series(dtype=object)).astype(str).values:
            self._show_error(f"{name} not found.")
            return
        out = people.copy()
        out.loc[out["name"] == name, "remaining_money"] = value
        storage.save_people(out)
        self.refresh_all()

    def on_remove_person(self):
        name = self.person_select_combo.currentText().strip()
        if not name:
            return
        people = storage.load_people()
        storage.save_people(people[people["name"].astype(str) != name].copy())
        self.refresh_all()


def run_gui() -> int:
    app = QtWidgets.QApplication.instance()
    created = False
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        created = True
    win = DraftToolWindow()
    win.show()
    if created:
        return app.exec()
    return 0

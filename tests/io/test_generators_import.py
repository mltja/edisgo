import pandas as pd
import numpy as np
import pytest
from shapely.geometry import Point

from edisgo import EDisGo
from edisgo.network.grids import LVGrid
from edisgo.io import generators_import


class TestGeneratorsImportWithoutTimeSeries:

    @classmethod
    def setup_class(self):
        self.edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_path,
            worst_case_analysis="worst-case"
        )

    def test_add_and_connect_mv_generator(self):

        # test voltage level 4
        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        generators_before = self.edisgo.topology.generators_df

        # add generator
        x = self.edisgo.topology.buses_df.at[
            "Bus_GeneratorFluctuating_2", "x"]
        y = self.edisgo.topology.buses_df.at[
            "Bus_GeneratorFluctuating_2", "y"]
        geom = Point((x, y))
        test_gen = pd.Series(
            {"electrical_capacity": 2.5,
             "geom": str(geom),
             "generation_type": "solar",
             "generation_subtype": "roof",
             "w_id": self.edisgo.topology.generators_df.at[
                 "GeneratorFluctuating_2", "weather_cell_id"],
             "voltage_level": 4},
            name=12345
        )

        comp_name = generators_import.add_and_connect_mv_generator(
            self.edisgo, test_gen)

        # check if number of buses increased
        assert len(buses_before) + 1 == len(self.edisgo.topology.buses_df)
        # check if number of lines increased
        assert len(lines_before) + 1 == len(self.edisgo.topology.lines_df)
        # check if number of generators increased
        assert len(generators_before) + 1 == len(
            self.edisgo.topology.generators_df)

        # check new bus
        new_bus = self.edisgo.topology.generators_df.at[comp_name, "bus"]
        assert self.edisgo.topology.buses_df.at[new_bus, "v_nom"] == 20
        # check new line
        new_line_df = self.edisgo.topology.get_connected_lines_from_bus(
            new_bus)
        assert len(new_line_df) == 1
        # check new generator
        assert self.edisgo.topology.generators_df.at[
                   comp_name, "p_nom"] == 2.5

        # test voltage level 5 (new bus needed)
        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        generators_before = self.edisgo.topology.generators_df

        # add generator
        x = self.edisgo.topology.buses_df.at[
            "Bus_GeneratorFluctuating_2", "x"]
        y = self.edisgo.topology.buses_df.at[
            "Bus_GeneratorFluctuating_6", "y"]
        geom = Point((x, y))
        test_gen = pd.Series(
            {"electrical_capacity": 2.5,
             "geom": str(geom),
             "generation_type": "solar",
             "generation_subtype": "roof",
             "w_id": self.edisgo.topology.generators_df.at[
                 "GeneratorFluctuating_2", "weather_cell_id"],
             "voltage_level": 5},
            name=123456
        )

        comp_name = generators_import.add_and_connect_mv_generator(
            self.edisgo, test_gen)

        # check if number of buses increased (by two because closest connection
        # object is a line)
        assert len(buses_before) + 2 == len(self.edisgo.topology.buses_df)
        # check if number of lines increased
        assert len(lines_before) + 2 == len(self.edisgo.topology.lines_df)
        # check if number of generators increased
        assert len(generators_before) + 1 == len(
            self.edisgo.topology.generators_df)

        # check new bus
        new_bus = self.edisgo.topology.generators_df.at[comp_name, "bus"]
        assert self.edisgo.topology.buses_df.at[new_bus, "v_nom"] == 20
        # check new line
        new_line_df = self.edisgo.topology.get_connected_lines_from_bus(
            new_bus)
        assert len(new_line_df) == 1
        assert "Bus_Generator_123456" in list(
            new_line_df.loc[new_line_df.index[0], ["bus0", "bus1"]])
        # check new generator
        assert self.edisgo.topology.generators_df.at[
                   comp_name, "p_nom"] == 2.5

        # test voltage level 5 (no new bus needed)
        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        generators_before = self.edisgo.topology.generators_df

        # add generator
        x = self.edisgo.topology.buses_df.at[
            "Bus_GeneratorFluctuating_6", "x"]
        y = self.edisgo.topology.buses_df.at[
            "Bus_GeneratorFluctuating_6", "y"]
        geom = Point((x, y))
        test_gen = pd.Series(
            {"electrical_capacity": 2.5,
             "geom": str(geom),
             "generation_type": "solar",
             "generation_subtype": "roof",
             "w_id": self.edisgo.topology.generators_df.at[
                 "GeneratorFluctuating_2", "weather_cell_id"],
             "voltage_level": 5},
            name=123456
        )

        comp_name = generators_import.add_and_connect_mv_generator(
            self.edisgo, test_gen)

        # check if number of buses increased (by two because closest connection
        # object is a line)
        assert len(buses_before) == len(self.edisgo.topology.buses_df)
        # check if number of lines increased
        assert len(lines_before) == len(self.edisgo.topology.lines_df)
        # check if number of generators increased
        assert len(generators_before) + 1 == len(
            self.edisgo.topology.generators_df)

        # check new generator
        assert self.edisgo.topology.generators_df.at[
                   comp_name, "p_nom"] == test_gen["electrical_capacity"]

    def test_add_and_connect_lv_generator(self):

        # test non-existent substation ID

        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        generators_before = self.edisgo.topology.generators_df

        # add generator
        test_gen = pd.Series(
            {"electrical_capacity": 0.3,
             "generation_type": "solar",
             "generation_subtype": "roof",
             "w_id": self.edisgo.topology.generators_df.at[
                 "GeneratorFluctuating_2", "weather_cell_id"],
             "voltage_level": 6,
             "mvlv_subst_id": 10},
            name=23456
        )

        comp_name = generators_import.add_and_connect_lv_generator(
            self.edisgo, test_gen)

        # check if number of buses stayed the same
        assert len(buses_before) == len(self.edisgo.topology.buses_df)
        # check if number of lines stayed the same
        assert len(lines_before) == len(self.edisgo.topology.lines_df)
        # check if number of generators increased
        assert len(generators_before) + 1 == len(
            self.edisgo.topology.generators_df)

        # check that new generator is connected to HV/MV station
        assert self.edisgo.topology.generators_df.at[
                   comp_name, "bus"] == "Bus_MVStation_1"

        # test missing substation ID

        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        generators_before = self.edisgo.topology.generators_df

        # add generator
        test_gen = pd.Series(
            {"electrical_capacity": 0.3,
             "generation_type": "solar",
             "generation_subtype": "roof",
             "w_id": self.edisgo.topology.generators_df.at[
                 "GeneratorFluctuating_2", "weather_cell_id"],
             "voltage_level": 6,
             "mvlv_subst_id": None},
            name=23456
        )

        comp_name = generators_import.add_and_connect_lv_generator(
            self.edisgo, test_gen)

        # check if number of buses stayed the same
        assert len(buses_before) == len(self.edisgo.topology.buses_df)
        # check if number of lines stayed the same
        assert len(lines_before) == len(self.edisgo.topology.lines_df)
        # check if number of generators increased
        assert len(generators_before) + 1 == len(
            self.edisgo.topology.generators_df)

        # check that new generator is connected to random substation
        new_bus = self.edisgo.topology.generators_df.at[comp_name, "bus"]
        assert self.edisgo.topology.buses_df.at[new_bus, "v_nom"] == 0.4
        lv_grid_id = self.edisgo.topology.buses_df.at[new_bus, "lv_grid_id"]
        lv_grid = LVGrid(id=lv_grid_id, edisgo_obj=self.edisgo)
        assert new_bus == lv_grid.station.index[0]
        # check new generator
        assert self.edisgo.topology.generators_df.at[
                   comp_name, "p_nom"] == 0.3

        # test existing substation ID (voltage level 6)

        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        generators_before = self.edisgo.topology.generators_df

        # add generator
        x = self.edisgo.topology.buses_df.at[
            "Bus_GeneratorFluctuating_6", "x"]
        y = self.edisgo.topology.buses_df.at[
            "Bus_GeneratorFluctuating_6", "y"]
        geom = Point((x, y))
        test_gen = pd.Series(
            {"electrical_capacity": 0.3,
             "geom": str(geom),
             "generation_type": "solar",
             "generation_subtype": "roof",
             "w_id": self.edisgo.topology.generators_df.at[
                 "GeneratorFluctuating_2", "weather_cell_id"],
             "voltage_level": 6,
             "mvlv_subst_id": 6},
            name=3456
        )

        comp_name = generators_import.add_and_connect_lv_generator(
            self.edisgo, test_gen)

        # check that number of buses increased
        assert len(buses_before) + 1 == len(self.edisgo.topology.buses_df)
        # check that number of lines increased
        assert len(lines_before) + 1 == len(self.edisgo.topology.lines_df)
        # check that number of generators increased
        assert len(generators_before) + 1 == len(
            self.edisgo.topology.generators_df)

        # check new bus
        new_bus = self.edisgo.topology.generators_df.at[comp_name, "bus"]
        assert self.edisgo.topology.buses_df.at[new_bus, "v_nom"] == 0.4
        # check new line
        new_line_df = self.edisgo.topology.get_connected_lines_from_bus(
            new_bus)
        assert len(new_line_df) == 1
        assert "Bus_Generator_3456" in list(
            new_line_df.loc[new_line_df.index[0], ["bus0", "bus1"]])
        lv_grid = LVGrid(id=6, edisgo_obj=self.edisgo)
        assert lv_grid.station.index[0] in list(
            new_line_df.loc[new_line_df.index[0], ["bus0", "bus1"]])
        # check new generator
        assert self.edisgo.topology.generators_df.at[
                   comp_name, "p_nom"] == 0.3
        assert comp_name in lv_grid.generators_df.index

        # test existing substation ID (voltage level 7)

        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        generators_before = self.edisgo.topology.generators_df

        # add generator
        test_gen = pd.Series(
            {"electrical_capacity": 0.03,
             "geom": str(geom),
             "generation_type": "solar",
             "generation_subtype": "roof",
             "w_id": self.edisgo.topology.generators_df.at[
                 "GeneratorFluctuating_2", "weather_cell_id"],
             "voltage_level": 7,
             "mvlv_subst_id": 1},
            name=3456
        )

        comp_name = generators_import.add_and_connect_lv_generator(
            self.edisgo, test_gen)

        # check that number of buses stayed the same
        assert len(buses_before) == len(self.edisgo.topology.buses_df)
        # check that number of lines stayed the same
        assert len(lines_before) == len(self.edisgo.topology.lines_df)
        # check that number of generators increased
        assert len(generators_before) + 1 == len(
            self.edisgo.topology.generators_df)

        # check bus
        gen_bus = self.edisgo.topology.generators_df.at[comp_name, "bus"]
        assert gen_bus == "Bus_Load_residential_LVGrid_1_4"
        assert self.edisgo.topology.buses_df.at[
                   gen_bus, "lv_grid_id"] == 1
        # check new generator
        assert self.edisgo.topology.generators_df.at[
                   comp_name, "p_nom"] == 0.03

        # ToDo test other options when connected to voltage level 7

    def test_update_grids(self):

        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        generators_before = self.edisgo.topology.generators_df

        # mv generators
        x = self.edisgo.topology.buses_df.at[
            "Bus_GeneratorFluctuating_6", "x"]
        y = self.edisgo.topology.buses_df.at[
            "Bus_GeneratorFluctuating_6", "y"]
        geom_gen_new = Point((x, y))
        generators_mv = pd.DataFrame(
            data={
                "geom": [None, None, str(geom_gen_new)],
                "electrical_capacity": [3.0, 2.67, 2.5],
                "generation_type": ["wind", "solar", "solar"],
                "generation_subtype": ["wind", "solar", "solar"],
                "w_id": [1122074, 1122075, 1122074],
                "voltage_level": [4, 4, 4]
            },
            index=[2, 3, 345]
        )
        generators_lv = pd.DataFrame(
            data={
                "geom": [None, None, str(geom_gen_new)],
                "electrical_capacity": [0.027, 0.005, 0.3],
                "generation_type": ["solar", "solar", "solar"],
                "generation_subtype": ["solar", "solar", "roof"],
                "w_id": [1122075, 1122075, 1122074],
                "voltage_level": [6, 6, 6],
                "mvlv_subst_id": [None, None, 6]
            },
            index=[13, 14, 456]
        )

        generators_import.update_grids(
            self.edisgo, generators_mv, generators_lv)

        # check number of generators
        assert len(self.edisgo.topology.generators_df) == 6
        assert len(self.edisgo.topology.mv_grid.generators_df) == 3

        # check removed generators
        assert "Generator_1" not in self.edisgo.topology.generators_df.index
        assert "GeneratorFluctuating_12" not in self.edisgo.topology.generators_df.index

        # check updated generators
        assert self.edisgo.topology.generators_df.at[
                   "GeneratorFluctuating_2", "p_nom"] == 3
        assert self.edisgo.topology.generators_df.at[
                   "GeneratorFluctuating_2", "subtype"] == "wind_wind_onshore"
        assert self.edisgo.topology.generators_df.at[
                   "GeneratorFluctuating_13", "p_nom"] == 0.027
        assert self.edisgo.topology.generators_df.at[
                   "GeneratorFluctuating_13", "subtype"] == "solar_solar_roof_mounted"

        # check generators that stayed the same
        assert self.edisgo.topology.generators_df.at[
                   "GeneratorFluctuating_3", "p_nom"] == 2.67
        assert self.edisgo.topology.generators_df.at[
                   "GeneratorFluctuating_3", "subtype"] == "solar_solar_ground_mounted"
        assert self.edisgo.topology.generators_df.at[
                   "GeneratorFluctuating_14", "p_nom"] == 0.005
        assert self.edisgo.topology.generators_df.at[
                   "GeneratorFluctuating_14", "subtype"] == "solar_solar_roof_mounted"

        # check new generators
        assert self.edisgo.topology.generators_df.at[
                   "Generator_solar_MVGrid_1_345", "p_nom"] == 2.5
        assert self.edisgo.topology.generators_df.at[
                   "Generator_solar_MVGrid_1_345", "type"] == "solar"
        assert self.edisgo.topology.generators_df.at[
                   "Generator_solar_LVGrid_6_456", "p_nom"] == 0.3
        assert self.edisgo.topology.generators_df.at[
                   "Generator_solar_LVGrid_6_456", "type"] == "solar"

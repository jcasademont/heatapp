max_x = 40
max_y = 40

racks_layout = \
    {'eo6': (5.5, 9.7), 'e9': (9.2, 9.7), 'e12': (12.5, 9.7), 'e15': (15.7, 9.7),
     'e20': (21, 9.7),
     'h3': (2.7, 7.5), 'h6': (5.5, 7.5), 'h9': (9.2, 7.5), 'h12': (12.5, 7.5),
     'h15': (15.47, 7.5), 'h19': (20, 7.5), 'h22': (23, 7.5), 'h25': (26.25, 7.5),
     'k1': (0.5, 5), 'k4': (3.5, 5), 'k8': (8, 5), 'k9': (9.2, 5),
     'k11': (10.5, 5), 'k12': (12.5, 5), 'k15': (15.7, 5), 'k19': (20, 5),
     'k22': (23, 5), 'k25': (26.25, 5),
     'n1': (0.25, 3), 'n2': (2, 3), 'n5': (5, 3), 'n10': (10, 3),
     'n14': (14.5, 3), 'n19': (20, 3), 'n22': (23, 3), 'n25': (26.25, 3),
     'q1': (1, 0.5), 'q4a': (4, 0.5), 'q8': (8, 0.5), 'q11': (11, 0.5),
     'q13': (13.5, 0.5), 'q20': (21, 0.5), 'q23': (24, 0.5),
     'q25': (26.5, 0.5)}

datacenter_layout = racks_layout.copy()
datacenter_layout.update(
    {'ahu_1_outlet': (4, 12.5), 'ahu_2_outlet': (10, 12.5),
     'ahu_3_outlet': (3, -2.5), 'ahu_4_outlet': (10, -2.5)})

datacenter_inlets_layout = datacenter_layout.copy()
datacenter_inlets_layout.update(
    {'ahu_1_air_on': (5, 13), 'ahu_2_air_on': (11, 13),
     'ahu_3_air_on': (5, -1), 'ahu_4_air_on': (11, -1)})

full_layout = datacenter_inlets_layout.copy()
full_layout.update(
    {'room_it_power_(kw)': (30, 30)})

ahu_layout = \
    {'ahu_1_air_on': (5, 13), 'ahu_2_air_on': (11, 13),
     'ahu_3_air_on': (5, -1), 'ahu_4_air_on': (11, -1),
     'ahu_1_outlet': (5, 12), 'ahu_2_outlet': (11, 12),
     'ahu_3_outlet': (5, 0), 'ahu_4_outlet': (11, 0),
     'ahu_1_power': (5, 12), 'ahu_2_power': (11, 12),
     'ahu_3_power': (5, 0), 'ahu_4_power': (11, 0),
     'ahu_1_inlet': (5, 12), 'ahu_2_inlet': (11, 12),
     'ahu_3_inlet': (5, 0), 'ahu_4_inlet': (11, 0),
     'ahu_1_inlet_rh': (5, 12), 'ahu_2_inlet_rh': (11, 12),
     'ahu_3_inlet_rh': (5, 0), 'ahu_4_inlet_rh': (11, 0), 'room_cooling_power_(kw)': (0,0), 'acu_supply_temperature_(c)': (0,0), 'acu_return_temperature_(c)': (0,0)}

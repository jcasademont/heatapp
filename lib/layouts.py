max_x = 40
max_y = 40

racks_layout = \
    {'eo6': (6, 10), 'e9': (9, 10), 'e12': (12, 10), 'e15': (15, 10),
     'e20': (20, 10),
     'h3': (3, 8), 'h6': (5, 8), 'h9': (9, 8), 'h12': (12, 8),
     'h15': (15, 8), 'h19': (19, 8), 'h22': (22, 8), 'h25': (25, 8),
     'k1': (1, 6), 'k4': (4, 6), 'k8': (8, 6), 'k9': (9, 6),
     'k11': (11, 6), 'k12': (12, 6), 'k15': (15, 6), 'k19': (19, 6),
     'k22': (22, 6), 'k25': (25, 6),
     'n1': (1, 4), 'n2': (2, 4), 'n5': (5, 4), 'n10': (10, 4),
     'n14': (14, 4), 'n19': (19, 4), 'n22': (22, 4), 'n25': (25, 4),
     'q1': (1, 2), 'q4a': (4, 2), 'q8': (8, 2), 'q11': (11, 2),
     'q13': (13, 2), 'q20': (20, 2), 'q20': (20, 2), 'q23': (23, 2),
     'q25': (25, 2)}

datacenter_layout = racks_layout.copy()
datacenter_layout.update(
    {'ahu_1_outlet': (5, 12), 'ahu_2_outlet': (11, 12),
     'ahu_3_outlet': (5, 0), 'ahu_4_outlet': (11, 0)})

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

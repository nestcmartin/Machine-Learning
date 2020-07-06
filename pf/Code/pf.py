dt['sexo'][dt['sexo'] == 0] = 'mujer'
dt['sexo'][dt['sexo'] == 1] = 'varon'

dt['tipo_dolor'][dt['tipo_dolor'] == 1] = 'angina tipica'
dt['tipo_dolor'][dt['tipo_dolor'] == 2] = 'angina atipica'
dt['tipo_dolor'][dt['tipo_dolor'] == 3] = 'dolor no anginoso'
dt['tipo_dolor'][dt['tipo_dolor'] == 4] = 'asintomatico'

dt['glucemia'][dt['glucemia'] == 0] = '<120mg/ml'
dt['glucemia'][dt['glucemia'] == 1] = '>120mg/ml'

dt['ecg'][dt['ecg'] == 0] = 'normal'
dt['ecg'][dt['ecg'] == 1] = 'anomalia onda ST-T'
dt['ecg'][dt['ecg'] == 2] = 'hipertrofia ventriculo izdo'

dt['angina_ejercicio'][dt['angina_ejercicio'] == 0] = 'no'
dt['angina_ejercicio'][dt['angina_ejercicio'] == 1] = 'si'

dt['pendiente_st'][dt['pendiente_st'] == 1] = 'creciente'
dt['pendiente_st'][dt['pendiente_st'] == 2] = 'plana'
dt['pendiente_st'][dt['pendiente_st'] == 3] = 'decreciente'

dt['talasemia'][dt['talasemia'] == 1] = 'normal'
dt['talasemia'][dt['talasemia'] == 2] = 'defecto fijo'
dt['talasemia'][dt['talasemia'] == 3] = 'defecto permanente'
domain: "[M] -> { S1[i0, i1, i2] : i0 <= M and i1 >= 1 and i1 <= M and i2 >= 1 and i2 <= i0 }"
child:
  context: "[M] -> { [] }"
  child:
    schedule: "[M] -> [{ S1[i0, i1, i2] -> [(i0)] }]"
    options: "[M] -> { separate[i0] }"
    child:
      schedule: "[M] -> [{ S1[i0, i1, i2] -> [(i1 + i2)] }]"
      options: "[M] -> { separate[i0] }"
      child:
        schedule: "[M] -> [{ S1[i0, i1, i2] -> [(i1)] }]"
        options: "[M] -> { separate[i0] }"

domain: "[M] -> { S1[i0, i1, i2, i3] : i0 >= 1 and i0 <= M and i1 >= 1 and i1 <= M and i2 >= 1 and i2 <= M and i3 >= 1 and i3 <= M }"
child:
  context: "[M] -> { [] }"
  child:
    schedule: "[M] -> [{ S1[i0, i1, i2, i3] -> [(i2)] }]"
    options: "[M] -> { separate[i0] }"
    child:
      schedule: "[M] -> [{ S1[i0, i1, i2, i3] -> [(i0)] }]"
      options: "[M] -> { separate[i0] }"
      child:
        schedule: "[M] -> [{ S1[i0, i1, i2, i3] -> [(i1)] }]"
        options: "[M] -> { separate[i0] }"
        child:
          schedule: "[M] -> [{ S1[i0, i1, i2, i3] -> [(i3)] }]"
          options: "[M] -> { separate[i0] }"

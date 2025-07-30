domain: "[M] -> { S1[i0, i1] : i0 >= 1 and i0 <= M and i1 >= 1 and i1 <= M; S2[i0, i1, i2] : i0 >= 1 and i0 <= M and i1 >= 1 and i1 <= M and i2 >= 1 and i2 <= M }"
child:
  context: "[M] -> { [] }"
  child:
    sequence:
    - filter: "[M] -> { S1[i0, i1] }"
      child:
        schedule: "[M] -> [{ S1[i0, i1] -> [(i0)] }]"
        options: "[M] -> { separate[i0] }"
        child:
          schedule: "[M] -> [{ S1[i0, i1] -> [(i1)] }]"
          options: "[M] -> { separate[i0] }"
    - filter: "[M] -> { S2[i0, i1, i2] }"
      child:
        schedule: "[M] -> [{ S2[i0, i1, i2] -> [(i0)] }]"
        options: "[M] -> { separate[i0] }"
        child:
          schedule: "[M] -> [{ S2[i0, i1, i2] -> [(i1)] }]"
          options: "[M] -> { separate[i0] }"
          child:
            schedule: "[M] -> [{ S2[i0, i1, i2] -> [(i2)] }]"
            options: "[M] -> { separate[i0] }"

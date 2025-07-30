domain: "[M] -> { S1[i0, i1] : i0 >= 1 and i0 <= M and i1 >= 1 and i1 <= M; S2[i0, i1] : i0 >= 1 and i0 <= M and i1 >= 1 and i1 <= M }"
child:
  context: "[M] -> { [] : M >= 2 }"
  child:
    sequence:
    - filter: "[M] -> { S1[i0, i1] }"
      child:
        schedule: "[M] -> [{ S1[i0, i1] -> [(i0)] }]"
        options: "[M] -> { separate[i0] }"
        child:
          schedule: "[M] -> [{ S1[i0, i1] -> [(i1)] }]"
          options: "[M] -> { separate[i0] }"
    - filter: "[M] -> { S2[i0, i1] }"
      child:
        schedule: "[M] -> [{ S2[i0, i1] -> [(i0)] }]"
        options: "[M] -> { separate[i0] }"
        child:
          schedule: "[M] -> [{ S2[i0, i1] -> [(i1)] }]"
          options: "[M] -> { separate[i0] }"

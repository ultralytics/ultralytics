domain: "[M] -> { S2[0]; S1[i0] : i0 >= 0 and i0 <= M }"
child:
  context: "[M] -> { [] }"
  child:
    schedule: "[M] -> [{ S2[i0] -> [(i0)]; S1[i0] -> [(i0)] }]"
    options: "[M] -> { atomic[i0] }"
    child:
      sequence:
      - filter: "[M] -> { S1[i0] }"
      - filter: "[M] -> { S2[i0] }"

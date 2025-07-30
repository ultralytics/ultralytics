domain: "[M] -> { S3[i0] : i0 >= 0 and i0 <= M; S2[i0] : i0 >= 1 and i0 <= M; S1[i0] : i0 >= 0 and i0 <= M }"
child:
  context: "[M] -> { [] : M >= 1 }"
  child:
    sequence:
    - filter: "[M] -> { S1[i0] }"
      child:
        schedule: "[M] -> [{ S1[i0] -> [(i0)] }]"
        options: "[M] -> { separate[i0] }"
    - filter: "[M] -> { S2[i0] }"
      child:
        schedule: "[M] -> [{ S2[i0] -> [(i0)] }]"
        options: "[M] -> { separate[i0] }"
    - filter: "[M] -> { S3[i0] }"
      child:
        schedule: "[M] -> [{ S3[i0] -> [(i0)] }]"
        options: "[M] -> { separate[i0] }"

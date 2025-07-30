domain: "[M] -> { S4[i0] : i0 >= 0 and i0 <= 1023 and i0 <= 1024 + M; S5[i0] : i0 >= 0 and i0 <= 1023 and i0 >= 1025 + M; S3[i0] : i0 >= 0 and i0 <= 1023; S2[i0] : i0 >= 0 and i0 <= 1023 and i0 >= 1025 + M; S1[i0] : i0 >= 0 and i0 <= 1023 and i0 <= 1024 + M; S6[i0] : i0 >= 0 and i0 <= 1023 }"
child:
  context: "[M] -> { [] }"
  child:
    schedule: "[M] -> [{ S2[i0] -> [(-1)]; S4[i0] -> [(i0)]; S1[i0] -> [(-1)]; S3[i0] -> [(-1)]; S6[i0] -> [(i0)]; S5[i0] -> [(i0)] }, { S2[i0] -> [(i0)]; S4[i0] -> [(0)]; S1[i0] -> [(i0)]; S3[i0] -> [(i0)]; S6[i0] -> [(0)]; S5[i0] -> [(0)] }]"
    options: "[M] -> { separate[i0] }"
    child:
      sequence:
      - filter: "[M] -> { S4[i0]; S1[i0] }"
      - filter: "[M] -> { S5[i0]; S2[i0] }"
      - filter: "[M] -> { S3[i0]; S6[i0] }"

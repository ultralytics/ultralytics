domain: "{ S5[i] : i >= 0 and i <= 14; S6[i, j] : i >= 0 and i <= 14 and j >= 0 and j <= 9; S7[i] : i >= 0 and i <= 14; S4[i] : i >= 0 and i <= 2; S2[i, j] : i >= 0 and i <= 2 and j >= 0 and j <= 11 - i; S1[i] : i >= 0 and i <= 2; S3[i, j] : i >= 0 and i <= 2 and j >= 0 and j <= 11 - i }"
child:
  context: "{ [] }"
  child:
    sequence:
    - filter: "{ S4[i0]; S2[i0, i1]; S1[i0]; S3[i0, i1] }"
      child:
        schedule: "[{ S3[i0, i1] -> [(1 + 3i0)]; S2[i0, i1] -> [(1 + 3i0)]; S1[i0] -> [(3i0)]; S4[i0] -> [(2 + 3i0)] }, { S3[i0, i1] -> [(1 + 2i1)]; S2[i0, i1] -> [(2i1)]; S1[i0] -> [(0)]; S4[i0] -> [(0)] }]"
        options: "{ separate[i0] }"
    - filter: "{ S5[i0]; S6[i0, i1]; S7[i0] }"
      child:
        schedule: "[{ S6[i0, i1] -> [(1 + 3i0)]; S7[i0] -> [(2 + 3i0)]; S5[i0] -> [(3i0)] }, { S6[i0, i1] -> [(i1)]; S7[i0] -> [(0)]; S5[i0] -> [(0)] }]"
        options: "{ separate[i0] }"

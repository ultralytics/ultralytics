domain: "{ S2[i0, -7, 0] : i0 >= 2 and i0 <= 10; S4[1, 0, 0]; S6[i0, -9 + i0, 2] : i0 >= 2 and i0 <= 10; S1[10, i1, 4] : i1 >= 1 and i1 <= 10; S5[i0, i1, 3] : i1 <= -1 + i0 and i0 <= 10 and i1 >= 1; S7[1, 0, 0]; S8[i0, 0, 3] : i0 >= 1 and i0 <= 9; S3[i0, i1, 1] : i1 >= -7 and i0 <= 10 and i1 <= -9 + i0 }"
child:
  context: "{ [] }"
  child:
    schedule: "[{ S6[i0, i1, i2] -> [(i0)]; S8[i0, i1, i2] -> [(i0)]; S5[i0, i1, i2] -> [(i0)]; S4[i0, i1, i2] -> [(i0)]; S7[i0, i1, i2] -> [(i0)]; S3[i0, i1, i2] -> [(i0)]; S1[i0, i1, i2] -> [(i0)]; S2[i0, i1, i2] -> [(i0)] }, { S6[i0, i1, i2] -> [(i1)]; S8[i0, i1, i2] -> [(i1)]; S5[i0, i1, i2] -> [(i1)]; S4[i0, i1, i2] -> [(i1)]; S7[i0, i1, i2] -> [(i1)]; S3[i0, i1, i2] -> [(i1)]; S1[i0, i1, i2] -> [(i1)]; S2[i0, i1, i2] -> [(i1)] }, { S6[i0, i1, i2] -> [(i2)]; S8[i0, i1, i2] -> [(i2)]; S5[i0, i1, i2] -> [(i2)]; S4[i0, i1, i2] -> [(i2)]; S7[i0, i1, i2] -> [(i2)]; S3[i0, i1, i2] -> [(i2)]; S1[i0, i1, i2] -> [(i2)]; S2[i0, i1, i2] -> [(i2)] }]"
    options: "{ separate[i0] }"
    child:
      sequence:
      - filter: "{ S1[i0, i1, i2] }"
      - filter: "{ S2[i0, i1, i2] }"
      - filter: "{ S3[i0, i1, i2] }"
      - filter: "{ S4[i0, i1, i2] }"
      - filter: "{ S5[i0, i1, i2] }"
      - filter: "{ S6[i0, i1, i2] }"
      - filter: "{ S7[i0, i1, i2] }"
      - filter: "{ S8[i0, i1, i2] }"

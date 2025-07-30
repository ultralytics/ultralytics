domain: "{ S2[idx4, idx5, idx6] : 14idx5 <= -84 + idx4 and 14idx5 >= -120 + idx4 and idx6 >= 24 + idx5 and idx6 <= 48 + idx5 and idx5 >= -6 and idx5 <= 18 and idx6 <= 24 - 2idx5; S4[idx4, idx5, idx6] : 14idx6 <= -462 + 3idx4 + 14idx5 and 14idx6 >= -570 + 3idx4 + 14idx5 and idx6 <= idx5 and idx6 >= -12 + idx5 and idx6 >= 6 and idx6 <= 30 and 2idx6 >= 42 - idx5; S6[idx4, idx5, idx6] : 77idx6 >= 924 - 6idx4 + 77idx5 and 77idx6 <= 1140 - 6idx4 + 77idx5 and idx6 <= 12 + idx5 and idx6 >= 6 + idx5 and idx6 >= 6 and idx6 <= 30 and 5idx6 >= 42 + 2idx5; S1[idx4, idx5, idx6] : 14idx6 >= 672 - 3idx4 + 14idx5 and 14idx6 <= 780 - 3idx4 + 14idx5 and idx6 >= 24 + idx5 and idx6 <= 48 + idx5 and idx5 >= -6 and idx5 <= 18 and idx6 >= 30 - 2idx5; S5[idx4, idx5, idx6] : 14idx6 <= 42 + idx4 and 14idx6 >= 6 + idx4 and idx6 <= idx5 and idx6 >= -12 + idx5 and idx6 >= 6 and idx6 <= 30 and 2idx6 <= 36 - idx5; S7[idx4, idx5, idx6] : 21idx6 <= 84 + 2idx4 - 7idx5 and 21idx6 >= 12 + 2idx4 - 7idx5 and idx6 <= 12 + idx5 and idx6 >= 6 + idx5 and idx6 >= 6 and idx6 <= 30 and 5idx6 <= 36 + 2idx5; S8[idx4, idx5, idx6] : 14idx6 >= 546 - 3idx4 + 14idx5 and 14idx6 <= 654 - 3idx4 + 14idx5 and idx6 >= idx5 and idx6 <= 24 + idx5 and idx5 >= 0 and idx5 <= 24 and idx6 >= 30 - 2idx5; S3[idx4, idx5, idx6] : idx4 >= 0 and idx4 <= 36 and idx6 >= -8idx5 and idx6 <= 24 - 8idx5 and idx5 >= 0 and idx5 <= 24; S9[idx4, idx5, idx6] : 14idx5 <= -42 + idx4 and 14idx5 >= -78 + idx4 and idx6 >= idx5 and idx6 <= 24 + idx5 and idx5 >= 0 and idx5 <= 24 and idx6 <= 24 - 2idx5 and idx6 >= 6 - 2idx5; S10[idx4, idx5, idx6] : 7idx6 <= idx4 - 28idx5 and 7idx6 >= -36 + idx4 - 28idx5 and idx6 >= idx5 and idx6 <= 24 + idx5 and idx5 >= 0 and idx5 <= 24 and idx6 <= -2idx5 }"
child:
  context: "{ [] }"
  child:
    schedule: "[{ S6[idx4, idx5, idx6] -> [(idx4)]; S8[idx4, idx5, idx6] -> [(idx4)]; S5[idx4, idx5, idx6] -> [(idx4)]; S9[idx4, idx5, idx6] -> [(idx4)]; S4[idx4, idx5, idx6] -> [(idx4)]; S10[idx4, idx5, idx6] -> [(idx4)]; S7[idx4, idx5, idx6] -> [(idx4)]; S3[idx4, idx5, idx6] -> [(idx4)]; S1[idx4, idx5, idx6] -> [(idx4)]; S2[idx4, idx5, idx6] -> [(idx4)] }, { S6[idx4, idx5, idx6] -> [(idx5)]; S8[idx4, idx5, idx6] -> [(idx5)]; S5[idx4, idx5, idx6] -> [(idx5)]; S9[idx4, idx5, idx6] -> [(idx5)]; S4[idx4, idx5, idx6] -> [(idx5)]; S10[idx4, idx5, idx6] -> [(idx5)]; S7[idx4, idx5, idx6] -> [(idx5)]; S3[idx4, idx5, idx6] -> [(idx5)]; S1[idx4, idx5, idx6] -> [(idx5)]; S2[idx4, idx5, idx6] -> [(idx5)] }, { S6[idx4, idx5, idx6] -> [(idx6)]; S8[idx4, idx5, idx6] -> [(idx6)]; S5[idx4, idx5, idx6] -> [(idx6)]; S9[idx4, idx5, idx6] -> [(idx6)]; S4[idx4, idx5, idx6] -> [(idx6)]; S10[idx4, idx5, idx6] -> [(idx6)]; S7[idx4, idx5, idx6] -> [(idx6)]; S3[idx4, idx5, idx6] -> [(idx6)]; S1[idx4, idx5, idx6] -> [(idx6)]; S2[idx4, idx5, idx6] -> [(idx6)] }]"
    options: "{ separate[i0] }"
    child:
      sequence:
      - filter: "{ S1[idx4, idx5, idx6] }"
      - filter: "{ S2[idx4, idx5, idx6] }"
      - filter: "{ S3[idx4, idx5, idx6] }"
      - filter: "{ S4[idx4, idx5, idx6] }"
      - filter: "{ S5[idx4, idx5, idx6] }"
      - filter: "{ S6[idx4, idx5, idx6] }"
      - filter: "{ S7[idx4, idx5, idx6] }"
      - filter: "{ S8[idx4, idx5, idx6] }"
      - filter: "{ S9[idx4, idx5, idx6] }"
      - filter: "{ S10[idx4, idx5, idx6] }"

domain: "{ S2[i0, 9 - i0] : i0 <= 8 and i0 >= 4; S1[i0, i1] : i1 >= 6 - i0 and i0 >= 2 and i1 >= 3 and i1 <= 6 and i1 >= -1 + i0 }"
child:
  context: "{ [] }"
  child:
    schedule: "[{ S1[i0, i1] -> [(i0)]; S2[i0, i1] -> [(i0)] }, { S1[i0, i1] -> [(i1)]; S2[i0, i1] -> [(i1)] }]"
    options: "{ separate[i0] }"
    child:
      sequence:
      - filter: "{ S1[i0, i1] }"
      - filter: "{ S2[i0, i1] }"

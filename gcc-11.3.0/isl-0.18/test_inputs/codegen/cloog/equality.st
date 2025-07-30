domain: "{ S2[i0, 4] : i0 >= 0 and i0 <= 5; S1[i0, 2i0] : i0 >= 0 and i0 <= 5 }"
child:
  context: "{ [] }"
  child:
    schedule: "[{ S1[i0, i1] -> [(i0)]; S2[i0, i1] -> [(i0)] }, { S1[i0, i1] -> [(i1)]; S2[i0, i1] -> [(i1)] }]"
    options: "{ atomic[i0] }"
    child:
      sequence:
      - filter: "{ S1[i0, i1] }"
      - filter: "{ S2[i0, i1] }"

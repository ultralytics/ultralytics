domain: "{ S2[i0, i1] : 3i1 = i0 and i0 >= 3 and i0 <= 100; S1[27] }"
child:
  context: "{ [] }"
  child:
    schedule: "[{ S2[i0, i1] -> [(i0)]; S1[i0] -> [(i0)] }, { S2[i0, i1] -> [(i1)]; S1[i0] -> [(0)] }]"
    options: "{ separate[i0] }"

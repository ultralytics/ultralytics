domain: "{ S2[i0, i1] : i0 >= 1 and i0 <= 100 and i1 >= 1 and i1 <= 100; S1[i0, i1] : i0 >= 1 and i0 <= 100 and i1 >= 1 and i1 <= 100 }"
child:
  context: "{ [] }"
  child:
    schedule: "[{ S1[i0, i1] -> [(i0 - i1)]; S2[i0, i1] -> [(1 + i0 - i1)] }]"
    options: "{ separate[i0] }"
    child:
      schedule: "[{ S1[i0, i1] -> [(2i1)]; S2[i0, i1] -> [(-1 + 2i1)] }]"
      options: "{ separate[i0] }"
      child:
        sequence:
        - filter: "{ S1[i0, i1] }"
        - filter: "{ S2[i0, i1] }"

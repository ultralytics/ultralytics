domain: "{ S1[i0] : i0 >= 0 and i0 <= 10; S2[i0] : i0 >= 0 and i0 <= 10 }"
child:
  context: "{ [] }"
  child:
    schedule: "[{ S2[i0] -> [(i0)]; S1[i0] -> [(i0)] }]"
    options: "{ separate[i0] }"
    child:
      sequence:
      - filter: "{ S2[i0] }"
      - filter: "{ S1[i0] }"

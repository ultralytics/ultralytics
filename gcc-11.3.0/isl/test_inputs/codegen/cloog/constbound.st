domain: "{ S2[i0, i1, i2] : i1 >= 0 and i1 <= 9999 and i2 >= 0 and i2 <= i1 and i1 >= 25 + 50i0 and i1 <= 49 + 50i0; S1[i0, i1, i2] : i1 >= 0 and i1 <= 9999 and i2 >= 0 and i2 <= i1 and i1 >= 50i0 and i1 <= 24 + 50i0 }"
child:
  context: "{ [] }"
  child:
    schedule: "[{ S1[i0, i1, i2] -> [(i0)]; S2[i0, i1, i2] -> [(i0)] }]"
    options: "{ separate[i0] }"
    child:
      sequence:
      - filter: "{ S1[i0, i1, i2] }"
        child:
          schedule: "[{ S1[i0, i1, i2] -> [(i1)] }, { S1[i0, i1, i2] -> [(i2)] }]"
          options: "{ separate[i0] }"
      - filter: "{ S2[i0, i1, i2] }"
        child:
          schedule: "[{ S2[i0, i1, i2] -> [(i1)] }, { S2[i0, i1, i2] -> [(i2)] }]"
          options: "{ separate[i0] }"

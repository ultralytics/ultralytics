domain: "{ S1[]; S3[i0] : i0 >= 0 and i0 <= 1; S2[] }"
child:
  context: "{ [] }"
  child:
    schedule: "[{ S2[] -> [(1)]; S3[i0] -> [(i0)]; S1[] -> [(0)] }]"
    options: "{ separate[i0] }"

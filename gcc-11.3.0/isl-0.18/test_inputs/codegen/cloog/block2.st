domain: "{ S2[i0, 1] : i0 >= 0 and i0 <= 9; S1[i0, 1] : i0 >= 0 and i0 <= 9; S3[i0, 1] : i0 >= 0 and i0 <= 9 }"
child:
  context: "{ [] }"
  child:
    schedule: "[{ S3[i0, i1] -> [(i0)]; S1[i0, i1] -> [(i0)]; S2[i0, i1] -> [(i0)] }, { S3[i0, i1] -> [(i1)]; S1[i0, i1] -> [(i1)]; S2[i0, i1] -> [(i1)] }]"
    options: "{ separate[i0] }"
    child:
      sequence:
      - filter: "{ S1[i0, i1] }"
      - filter: "{ S3[i0, i1] }"
      - filter: "{ S2[i0, i1] }"

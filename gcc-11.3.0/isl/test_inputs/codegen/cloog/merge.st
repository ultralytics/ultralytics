domain: "{ S3[i0] : i0 >= 0 and i0 <= 10; S1[0]; S2[i0] : i0 >= 2 and i0 <= 10 }"
child:
  context: "{ [] }"
  child:
    schedule: "[{ S2[i0] -> [(i0)]; S3[i0] -> [(i0)]; S1[i0] -> [(i0)] }]"
    options: "{ atomic[i0] }"
    child:
      sequence:
      - filter: "{ S1[i0] }"
      - filter: "{ S2[i0] }"
      - filter: "{ S3[i0] }"

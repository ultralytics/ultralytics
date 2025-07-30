domain: "{ S1[j, div41, div42, 2, mod6_a] : 3mod6_a = -2 + j and j >= 1 and j <= 10 and 3div41 >= j and 3div42 >= -1 + j and 3div42 <= 1 + j and 3div41 <= 2 + j; S2[j, div41, div42, 2, mod6_a] : 3div42 = 1 + j and 3mod6_a = -2 + j and 3div41 >= 1 + j and 3div41 <= 2 + j and j >= 1 and j <= 10; S3[j, div41, div42, 2, mod6_a] : 3mod6_a = -2 + j and j >= 1 and j <= 10 and 3div41 >= j and 3div42 >= -1 + j and 3div42 <= 1 + j and 3div41 <= 2 + j }"
child:
  context: "{ [] }"
  child:
    schedule: "[{ S1[j, div41, div42, mod6, mod6_a] -> [(j)]; S3[j, div41, div42, mod6, mod6_a] -> [(j)]; S2[j, div41, div42, mod6, mod6_a] -> [(j)] }, { S1[j, div41, div42, mod6, mod6_a] -> [(div41)]; S3[j, div41, div42, mod6, mod6_a] -> [(div41)]; S2[j, div41, div42, mod6, mod6_a] -> [(div41)] }, { S1[j, div41, div42, mod6, mod6_a] -> [(div42)]; S3[j, div41, div42, mod6, mod6_a] -> [(div42)]; S2[j, div41, div42, mod6, mod6_a] -> [(div42)] }, { S1[j, div41, div42, mod6, mod6_a] -> [(mod6)]; S3[j, div41, div42, mod6, mod6_a] -> [(mod6)]; S2[j, div41, div42, mod6, mod6_a] -> [(mod6)] }, { S1[j, div41, div42, mod6, mod6_a] -> [(mod6_a)]; S3[j, div41, div42, mod6, mod6_a] -> [(mod6_a)]; S2[j, div41, div42, mod6, mod6_a] -> [(mod6_a)] }]"
    options: "{ separate[i0] }"
    child:
      sequence:
      - filter: "{ S1[j, div41, div42, mod6, mod6_a] }"
      - filter: "{ S2[j, div41, div42, mod6, mod6_a] }"
      - filter: "{ S3[j, div41, div42, mod6, mod6_a] }"

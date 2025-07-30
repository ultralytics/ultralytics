domain: "{ S2[j, a, b] : 2a = j and j >= 1 and j <= 10 and 2b <= j and 2b >= -1 + j; S1[j, a, b] : 2a = j and 2b = j and j <= 8 and j >= 2 }"
child:
  context: "{ [] }"
  child:
    schedule: "[{ S1[j, a, b] -> [(j)]; S2[j, a, b] -> [(j)] }, { S1[j, a, b] -> [(a)]; S2[j, a, b] -> [(a)] }, { S1[j, a, b] -> [(b)]; S2[j, a, b] -> [(b)] }]"
    options: "{ separate[i0] }"
    child:
      sequence:
      - filter: "{ S1[j, a, b] }"
      - filter: "{ S2[j, a, b] }"

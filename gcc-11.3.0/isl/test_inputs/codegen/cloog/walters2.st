domain: "{ S2[j, 51] : j <= 24 and j >= 1; S2[25, i] : i <= 51 and i >= 1; S2[j, 0] : j <= 25 and j >= 1; S2[0, i] : i <= 51 and i >= 0; S1[j, i] : j >= 1 and j <= 24 and i >= 1 and i <= 50 }"
child:
  context: "{ [] }"
  child:
    schedule: "[{ S1[j, i] -> [(j)]; S2[j, i] -> [(j)] }, { S1[j, i] -> [(i)]; S2[j, i] -> [(i)] }]"
    options: "{ separate[i0] }"
    child:
      sequence:
      - filter: "{ S1[j, i] }"
      - filter: "{ S2[j, i] }"

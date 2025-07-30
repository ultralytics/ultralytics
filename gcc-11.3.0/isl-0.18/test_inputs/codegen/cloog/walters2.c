{
  for (int c1 = 0; c1 <= 51; c1 += 1)
    S2(0, c1);
  for (int c0 = 1; c0 <= 24; c0 += 1) {
    S2(c0, 0);
    for (int c1 = 1; c1 <= 50; c1 += 1)
      S1(c0, c1);
    S2(c0, 51);
  }
  for (int c1 = 0; c1 <= 51; c1 += 1)
    S2(25, c1);
}

for (int c0 = 0; c0 <= 5; c0 += 1) {
  S1(c0, c0);
  for (int c1 = c0; c1 <= 5; c1 += 1)
    S2(c0, c1);
  S3(c0, 5);
}

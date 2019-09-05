using DelimitedFiles

function Laplacian_normalization(M)
    for i = 1:size(M, 1)
        M[i, i] = sum(M[i, :])
    end
    for i ∈ 1:size(M, 1)
        for j ∈ 1:size(M, 2)
            M[i, j] = M[i, j] / sqrt(M[i, i] * M[j, j])
        end
    end
    return M
end

function MBiRW(simR, simD, A, α, l, r)
    MR = Laplacian_normalization(simR)
    MD = Laplacian_normalization(simD)
    A = A / sum(A)
    RD = A
    for i ∈ 1:max(l, r)
        dflag = 1
        rflag = 1
        if i ≤ l
            Rr = α * RD * MR + (1 - α) * A
            rflag = 1
        end
        if i ≤ r
            Rd = α * MD * RD + (1 - α) * A
            dflag = 1
        end
        RD = (rflag * Rr + dflag * Rd) / (rflag + dflag)
    end
    return RD
end

simR = readdlm("./Datasets/DrugSimMat")
simD = readdlm("./Datasets/DiseaseSimMat")
A = readdlm("./Datasets/DiDrAMat")

# α = 0.3
l = 2
r = 2
# for α ∈ 0.1:0.1:0.9
#     RD = MBiRW(simR, simD, A, α, l, r)
#     writedlm(string(α) * "JlRDMat", RD)
# end

data=readdlm("./0.3JlRDMat")
println(sum(data))

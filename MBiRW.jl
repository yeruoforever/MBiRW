using DelimitedFiles
using Random
using Plots


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

function sampling_indexs(A::Matrix)
    h, w = size(A)
    vcat([shuffle(1:w)' for i ∈ 1:h]...)
end

function sampling_group_indexs(s::Matrix, folds, id)
    h, w = size(s)
    group_size = fld(w, folds)
    # if folds == id
    #     return @view s[:, (id-1)*group_size+1:end]
    # else
    #     return @view s[:, ((id-1)*group_size+1:id*group_size)]
    # end
    @view s[:, (id-1)*group_size+1:id*group_size]
end

function train_matrix(A::Matrix, samplings)
    m = copy(A)
    h, w = size(A)
    for i ∈ 1:h
        for j ∈ samplings[i, :]
            m[i, j] = 0
        end
    end
    m
end

function evalution_vector(outputs, labels, threshold)
    pair = zip(outputs, labels)
    tp = sum(pair) do (pre, target)
        target == 1 && pre >= threshold
    end
    fn = sum(pair) do (pre, target)
        target == 1 && pre < threshold
    end
    fp = sum(pair) do (pre, target)
        target == 0 && pre >= threshold
    end
    tn = sum(pair) do (pre, target)
        target == 0 && pre < threshold
    end
    [tp, fn, fp, tn]
end

function evalution_matrix(outputs, targets)
    thresholds = sort(outputs,rev=true)
    hcat([evalution_vector(outputs, targets, threshold) for threshold ∈ thresholds]...)
end

function PR_ROC(evalution::Matrix)
    TP, FN, FP, TN = 1:4
    p1 = plot(zeros(0),xlabel="Recall",ylabel="Precision")
    p2 = plot(zeros(0),xlabel="FPR",ylabel="TPR")
    AUC = 0
    current_x, current_y = 0, 0
    for i = 1:size(evalution, 2)
        ev = evalution[:, i]
        P = ev[TP] / (ev[TP] + ev[FP])
        R = ev[TP] / (ev[TP] + ev[FN])
        TPR = R
        FPR = ev[FP] / (ev[TN] + ev[FP])
        push!(p1, (R, P))
        push!(p2, (FPR, TPR))
        AUC += (FPR - current_x) * (current_y + TPR) / 2
        current_x, current_y = FPR, TPR
    end
    display(plot(p1, p2, layout = (1, 2), legend = false))
    AUC
end

function fold_cross(simR, simD, A, α, l, r, folds = 10)
    samplings = sampling_indexs(A)
    group_size = fld(size(A, 2), folds)
    ems = zeros(Int, 4, group_size * size(A, 1))
    for fold ∈ 1:folds
        sg = sampling_group_indexs(samplings, folds, fold)
        tm = train_matrix(A, sg)
        rw = MBiRW(simR, simD, tm, α, l, r)
        h, w = size(sg)
        outputs = [rw[i, j] for i ∈ 1:h for j ∈ sg[i, :]]
        targets = [A[i, j] for i ∈ 1:h for j ∈ sg[i, :]]
        em = evalution_matrix(outputs, targets)
        ems += em
    end
    PR_ROC(ems)
end

simR = readdlm("./Datasets/DrugSimMat")
simD = readdlm("./Datasets/DiseaseSimMat")
A = readdlm("./Datasets/DiDrAMat")
α = 0.3
l = 3
r = 3
AUC = fold_cross(simR, simD, A, α, l, r)
println(AUC)
